#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

layout ( local_size_x = 16, local_size_y = 16 ) in;

#include "common.h"

// the draw image
layout ( rgba16f, set = 0, binding = 1 ) uniform image2D image;

// the state image
layout ( set = 0, binding = 2 ) uniform sampler2D state;

vec3 TonemapUchimura2 ( vec3 v ) {
	const float P = 1.0;  // max display brightness
	const float a = 1.7;  // contrast
	const float m = 0.1;  // linear section start
	const float l = 0.0;  // linear section length
	const float c = 1.33; // black
	const float b = 0.0;  // pedestal

	float l0 = ( ( P - m ) * l ) / a;
	float L0 = m - m / a;
	float L1 = m + ( 1.0 - m ) / a;
	float S0 = m + l0;
	float S1 = m + a * l0;
	float C2 = ( a * P ) / ( P - S1 );
	float CP = -C2 / P;

	vec3 w0 = 1.0f - smoothstep( 0.0f, m, v );
	vec3 w2 = step( m + l0, v );
	vec3 w1 = 1.0f - w0 - w2;

	vec3 T = m * pow( v / m, vec3( c ) ) + vec3( b );
	vec3 S = P - ( P - S1 ) * exp( CP * ( v - S0 ) );
	vec3 L = m + a * ( v - vec3( m ) );

	return T * w0 + L * w1 + S * w2;
}

void main () {
	// Computing a UV for the texture sampling operation
	vec2 loc = ( gl_GlobalInvocationID.xy + vec2( 0.5f ) ) / GlobalData.accumulatorResolution.xy * ( vec2( GlobalData.accumulatorResolution ) / vec2( GlobalData.presentBufferResolution ) );

	vec3 color = GlobalData.brightnessScalar * texture( state, loc ).rgb;

	// Sample the image and store the result
	imageStore( image, ivec2( gl_GlobalInvocationID.xy ), vec4( TonemapUchimura2( color ), 1.0f ) );
}