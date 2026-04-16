#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

layout ( local_size_x = 16, local_size_y = 16 ) in;

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

void main () {
	// bounds checking
	ivec2 loc = ivec2( gl_GlobalInvocationID.xy );
	ivec2 size = ivec2( imageSize( massAtomic ).xy );
	if ( all( lessThan( loc, size ) ) ) {
	// normalization for the momentum accumulated on the grid
		// also apply forces like gravity ( mouse repulsion )
		// should the new quantities be written to new images? floating point/filtered...
		int vX = imageLoad( velocityXAtomic, loc ).r;
		int vY = imageLoad( velocityYAtomic, loc ).r;
		vec2 v = vec2( vX, vY ) / GlobalData.fixedPointScalar;

		int mass = imageLoad( massAtomic, loc ).r;
		if ( mass != 0 ) { // there has been some write to this cell
			// normalizing by dividing by the mass
			v /= ( float( mass ) / GlobalData.fixedPointScalar );

			// force of gravity
			v += GlobalData.dT * vec2( 0.0f, GlobalData.gravityScalar );

			// "slip" condition at the boundaries
			if ( loc.x < 2 || loc.x > ( size.x - 3 ) ) v.x = 0.0f;
			if ( loc.y < 2 || loc.y > ( size.y - 3 ) ) v.y = 0.0f;

			// clamping max velocity
			vec2 vCache = v;
			v = clamp( v, vec2( -1000.0f ), vec2( 1000.0f ) );
			if ( v != vCache ) {
				v = vec2( 0.0f );
			}

			// storing back floating point value of V
			imageStore( resolvedAtomics, loc, vec4( v.xyxy ) );
		}
	}
}