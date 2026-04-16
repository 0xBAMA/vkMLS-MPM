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
	int idx = int( gl_GlobalInvocationID.x );

	// operates on points, using information written to the mass grids prior + quadratic interpolation weights
	vec2 cellIdx = floor( points[ idx ].position );
	vec2 cellDiff = ( points[ idx ].position - cellIdx ) - 0.5f;
	vec2 weights[ 3 ];
	weights[ 0 ] = 0.5f * pow( 0.5f - cellDiff, vec2( 2.0f ) );
	weights[ 1 ] = 0.75f - pow( cellDiff, vec2( 2.0f ) );
	weights[ 2 ] = 0.5f * pow( 0.5f + cellDiff, vec2( 2.0f ) );

	// accumulate density over neighbouring 3x3 cells
	float density = 0.0f;
	for ( int gx = 0; gx < 3; ++gx ) {
		for ( int gy = 0; gy < 3; ++gy ) {
			float weight = weights[ gx ].x * weights[ gy ].y;
			density += ( imageLoad( massAtomic, ivec2( cellIdx.x + gx - 1, cellIdx.y + gy - 1 ) ).r / GlobalData.fixedPointScalar ) * weight;
		}
	}

	// per-particle volume estimate has now been computed
	points[ idx ].v0 = points[ idx ].mass / density;
}