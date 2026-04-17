//=========================================================
// push constants block - updated at smallest scope
layout( push_constant ) uniform constants {
// RNG seeding from the CPU
	uint wangSeed;

	float pointScale;
} PushConstants;

//=========================================================
// Global config etc data in a UBO
layout( set = 0, binding = 0 ) uniform globalData {
	// buffer resolutions:
	uvec2 presentBufferResolution;
	uvec2 accumulatorResolution;
	vec3 mouseLoc;

	int frameNumber;
	int reset;

	float brightnessScalar;
	float resolutionScalar;

	float gravityScalar;
	float fixedPointScalar;

	uint numPoints;
	uint numPointsFluid;

	float dT;

	// Lamé parameters for stress-strain relationship
	float elasticLambda;
	float elasticMu;

	// new fluid parameters for Tait compressible fluid equation of state
	float restDensity;
	float dynamicViscosity;
	float eosStiffness;
	float eosPower;

	float mouseSize;
	float mouseForceScalar;

	// nsight layout: vec2u; vec2; int; int; float; float; ...
} GlobalData;
//=========================================================

#ifndef saturate
#define saturate(x) clamp(x, 0, 1)
#endif

#ifndef UINT_MAX
#define UINT_MAX (0xFFFFFFFF-1)
#endif

#ifndef PI_DEFINED
#define PI_DEFINED
const float pi = 3.141592f;
const float tau = 2.0f * pi;
const float sqrtpi = 1.7724538509f;
#endif

#ifndef REMAP_DEFINED
#define REMAP_DEFINED
float remap ( float value, float inLow, float inHigh, float outLow, float outHigh ) {
	return outLow + ( value - inLow ) * ( outHigh - outLow ) / ( inHigh - inLow );
}
#endif