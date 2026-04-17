#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

layout ( local_size_x = 64 ) in;

#include "common.h"

struct point {
	vec2 position;
	vec2 velocity;

	mat2 C;
	mat2 Fs;

	float mass;
	float v0;

	int particleType;
	float pad;
};
layout ( set = 0, binding = 1, std430 ) buffer pointBuffer {
	point points[];
};

layout ( r32i, set = 0, binding = 2 ) uniform iimage2D velocityXAtomic;
layout ( r32i, set = 0, binding = 3 ) uniform iimage2D velocityYAtomic;
layout ( r32i, set = 0, binding = 4 ) uniform iimage2D massAtomic;
layout ( rg32f, set = 0, binding = 5 ) uniform image2D resolvedAtomics;

#include "random.h"
void main () {
	seed = PushConstants.wangSeed + gl_GlobalInvocationID.x * 8675309;

	const int idx = int( gl_GlobalInvocationID.x );

	// velocity is calculated from the grid each time
	points[ idx ].velocity = vec2( 0.0f );

	// quadratic interpolation weights
	ivec2 cellIdx = ivec2( points[ idx ].position );
	vec2 cellDiff = ( points[ idx ].position - cellIdx ) - 0.5f;
	vec2 weights[ 3 ] = vec2[ 3 ] (
		0.5f * pow( 0.5f - cellDiff, vec2( 2.0f ) ),
		0.75f - pow( cellDiff, vec2( 2.0f ) ),
		0.5f * pow( 0.5f + cellDiff, vec2( 2.0f ) )
	);

	// constructing affine per-particle momentum matrix from APIC / MLS-MPM.
	// see APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf), page 6
	// below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
	// where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
	mat2 B = mat2( 0.0f );
	for ( int gx = 0; gx < 3; ++gx ) {
		for ( int gy = 0; gy < 3; ++gy ) {
			float weight = weights[ gx ].x * weights[ gy ].y;
			ivec2 cellIdxInner = ivec2( cellIdx.x + gx - 1, cellIdx.y + gy - 1 );
			 vec2 dist = ( cellIdxInner - points[ idx ].position ) + 0.5f;

			vec2 velocity = imageLoad( resolvedAtomics, cellIdxInner ).xy;
			vec2 weightedVelocity = velocity * weight;

			// APIC paper equation 10, constructing inner term for B
			B += mat2( weightedVelocity * dist.x, weightedVelocity * dist.y );
			points[ idx ].velocity += weightedVelocity;
		}
	}
	points[ idx ].C = B * 4;

	// advect particles
	points[ idx ].position += points[ idx ].velocity * GlobalData.dT;

	// boundary condition
	points[ idx ].position = clamp( points[ idx ].position, vec2( 1.0f ), vec2( imageSize( massAtomic ).xy - 2 ) );

	// something to try: doesn't work with slip condition
//	const ivec2 iS = ivec2( imageSize( massAtomic ).xy );
//	if ( points[ idx ].position.x >= iS.x ) points[ idx ].position.x -= iS.x;
//	if ( points[ idx ].position.x < 0.0f ) points[ idx ].position.x += iS.x;
//	if ( points[ idx ].position.y >= iS.y ) points[ idx ].position.y -= iS.y;
//	if ( points[ idx ].position.y < 0.0f ) points[ idx ].position.y += iS.y;

	// mouse interaction
	if ( GlobalData.mouseLoc.z > 0.5f ) {
		vec2 dist = vec2( 1.0f, 1.0f ) * ( points[ idx ].position - GlobalData.mouseLoc.xy );
//		const float MouseSize = 65.0f;
		float MouseSize = GlobalData.mouseSize;

		if ( dot( dist, dist ) < MouseSize * MouseSize ) {
			float norm_factor = ( length( dist ) / MouseSize );
			norm_factor = pow( sqrt( norm_factor ), 10 );
			vec2 force = normalize( dist ) * norm_factor * GlobalData.mouseForceScalar;
			points[ idx ].velocity += force;
//			points[ idx ].velocity += GlobalData.mouseForceScalar * vec2( 0.1f, 0.05f * ( NormalizedRandomFloat() - 0.5f ) );
		}
	}
	if ( points[ idx ].particleType == 0 ) {
		// clamping max in the Fs matrix
		if (
			abs( points[ idx ].Fs[ 0 ][ 0 ] ) > 100000.0f ||
			abs( points[ idx ].Fs[ 1 ][ 0 ] ) > 100000.0f ||
			abs( points[ idx ].Fs[ 0 ][ 1 ] ) > 100000.0f ||
			abs( points[ idx ].Fs[ 1 ][ 1 ] ) > 100000.0f ) {
			points[ idx ].velocity = vec2( 0.0f );
			points[ idx ].C = mat2( 0.0f );
			points[ idx ].Fs = mat2( 1.0f );
		}

		// deformation gradient update - MPM course, equation 181
		// Fp' = (I + dt * p.C) * Fp
		mat2 Fp_new = mat2( 1.0f );
		Fp_new += GlobalData.dT * points[ idx ].C;
		points[ idx ].Fs = Fp_new * points[ idx ].Fs;
	}
}