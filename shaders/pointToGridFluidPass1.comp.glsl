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

		// for all surrounding 9 cells
		for ( uint gx = 0; gx < 3; ++gx ) {
			for ( uint gy = 0; gy < 3; ++gy ) {
				float weight = weights[ gx ].x * weights[ gy ].y;

				ivec2 cellIdxInner = ivec2( cellIdx.x + gx - 1, cellIdx.y + gy - 1 );
				vec2 cellDist = ( vec2( cellIdxInner ) - p.position ) + 0.5f;
				vec2 Q = p.C * cellDist;

				// MPM course, equation 172
				float weightedMass = weight * p.mass;
				imageAtomicAdd( massAtomic, cellIdxInner, int( weightedMass * GlobalData.fixedPointScalar ) );

				// velocity grid contribution...
				// APIC P2G momentum contribution only, this stage - momentum comes later
				vec2 writeV = weightedMass * ( p.velocity + Q );

				// fixed point adjustment applied on write and in reverse on read
				imageAtomicAdd( velocityXAtomic, cellIdxInner, int( writeV.x * GlobalData.fixedPointScalar ) );
				imageAtomicAdd( velocityYAtomic, cellIdxInner, int( writeV.y * GlobalData.fixedPointScalar ) );
			}
		}
	}
}