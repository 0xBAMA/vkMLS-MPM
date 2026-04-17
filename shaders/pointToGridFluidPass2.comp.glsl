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

void main () {
	int idx = int( gl_GlobalInvocationID );
	point p = points[ idx + GlobalData.numPoints ];

	// points make atomic writes on the buffers
		// ( momentum, mass )

	if ( p.particleType == 1 ) { // fluid particle

		// quadratic interpolation weights
		uvec2 cellIdx = uvec2( p.position );
		vec2 cellDiff = ( p.position - cellIdx ) - 0.5f;

		vec2 weights[3];
		weights[0] = 0.5f * pow( 0.5f - cellDiff, vec2( 2.0f ) );
		weights[1] = 0.75f - pow( cellDiff, vec2( 2.0f ) );
		weights[2] = 0.5f * pow( 0.5f + cellDiff, vec2( 2.0f ) );

		// first thing is to solve for the volume, from a local estimate of the density
		float density = 0.0f;
		for ( int gx = 0; gx < 3; ++gx ) {
			for ( int gy = 0; gy < 3; ++gy ) {
				float weight = weights[ gx ].x * weights[ gy ].y;
				density += float( imageLoad( massAtomic, ivec2( cellIdx ) + ivec2( gx - 1, gy - 1 ) ).r ) / GlobalData.fixedPointScalar;
			}
		}
		float volume = p.mass / density;

		// then we are applying the Tait equation of state... clamped by the original author as "a bit of a hack"...
			// they say clamping helps prevent particles absorbing into each other with negative pressures

		// end goal, constitutive equation for isotropic fluid:
		// stress = -pressure * I + viscosity * (velocity_gradient + velocity_gradient_transposed)

		float pressure = max( -0.1f, GlobalData.eosStiffness * ( pow( density / GlobalData.restDensity, GlobalData.eosPower ) - 1.0f ) );
		mat2 stress = mat2(
			-pressure, 0,
			0, -pressure
		);

		// velocity gradient - CPIC eq. 17, where deriv of quadratic polynomial is linear
		mat2 dudv = p.C;
		mat2 strain = dudv;

		float trace = strain[1][0] + strain[0][1];
		strain[0][1] = trace;
		strain[1][0] = trace;

		mat2 viscosityTerm = GlobalData.dynamicViscosity * strain;
		stress += viscosityTerm;

		mat2 eq_16_term_0 = -volume * 4 * stress * GlobalData.dT;

		// for all surrounding 9 cells
		for ( uint gx = 0; gx < 3; ++gx ) {
			for ( uint gy = 0; gy < 3; ++gy ) {
				float weight = weights[ gx ].x * weights[ gy ].y;

				ivec2 cellIdxInner = ivec2( cellIdx.x + gx - 1, cellIdx.y + gy - 1 );
				vec2 cellDist = ( vec2( cellIdxInner) - p.position ) + 0.5f;

				// then there is a fused force and momentum force applied
				vec2 writeV = eq_16_term_0 * ( weight * cellDist );

				// fixed point adjustment applied on write and in reverse on read
				imageAtomicAdd( velocityXAtomic, cellIdxInner, int( writeV.x * GlobalData.fixedPointScalar ) );
				imageAtomicAdd( velocityYAtomic, cellIdxInner, int( writeV.y * GlobalData.fixedPointScalar ) );
			}
		}
	}
}