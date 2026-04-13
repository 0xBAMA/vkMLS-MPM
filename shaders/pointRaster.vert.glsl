#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "common.h"
#include "random.h"

struct point {
	vec2 position;
	vec2 velocity;
};

layout( set = 0, binding = 1, std430 ) readonly buffer pointBuffer {
	point points[];
};

layout ( location = 0 ) out flat uint index;

void main () {
	vec2 center = ( 2.0f * points[ gl_VertexIndex ].position.xy / GlobalData.accumulatorResolution.xy ) - vec2( 1.0f );
	center.x *= GlobalData.accumulatorResolution.x / GlobalData.accumulatorResolution.y;

	// defaulting
	gl_PointSize = 1.0f;

	// writing the point locations
	gl_Position = vec4( center, 0.0f, 1.0f );
}