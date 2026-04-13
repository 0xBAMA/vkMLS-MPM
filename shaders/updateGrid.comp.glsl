#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

layout ( local_size_x = 16, local_size_y = 16 ) in;

#include "common.h"

struct point {
	vec2 position;
	vec2 velocity;
};
layout ( set = 0, binding = 1, std430 ) buffer pointBuffer {
	point points[];
};

layout ( r32i, set = 0, binding = 2 ) uniform iimage2D velocityXAtomic;
layout ( r32i, set = 0, binding = 3 ) uniform iimage2D velocityYAtomic;
layout ( r32i, set = 0, binding = 4 ) uniform iimage2D massAtomic;

void main () {
	ivec2 loc = ivec2( gl_GlobalInvocationID.xy );
	ivec2 size = ivec2( imageSize( massAtomic ).xy );
	// bounds checking
	if ( all( lessThan( loc, size ) ) ) {

		// normalization for the momentum accumulated on the grid
		// also apply forces like gravity ( mouse repulsion )
			// should the new quantities be written to new images? floating point/filtered...

		// imageStore( velocityXAtomic, loc, ivec4( 0 ) );
		// imageStore( velocityYAtomic, loc, ivec4( 0 ) );
		// imageStore( massAtomic, loc, ivec4( 0 ) );
	}
}