#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

layout ( local_size_x = 64 ) in;

#include "common.h"
#include "random.h"
#include "hg_sdf.h"

struct raySegment {
	float wavelength;
	float brightness;
	vec2 a;	// first point
	vec2 b;	// second point
};

raySegment getDefaultSegment () {
	raySegment r;
	r.wavelength = 0.0f;
	r.brightness = 0.0f;
	r.a = r.b = vec2( 0.0f );
	return r;
}

layout( set = 0, binding = 1, std430 ) buffer rayBuffer {
	raySegment rays[];
};

layout( set = 0, binding = 2 ) uniform sampler2D iCDFBuffer;
float getWavelengthForLight( uint selectedLight ) {
	return texture( iCDFBuffer, vec2( NormalizedRandomFloat(), ( selectedLight + 0.5f ) / textureSize( iCDFBuffer, 0 ).y ) ).r;
}

layout( set = 0, binding = 3 ) uniform usampler2D pickBuffer;
uint getPickedLight() {
	return texture( pickBuffer, vec2( NormalizedRandomFloat(), NormalizedRandomFloat() ) ).r;
}

struct LightEmitterParameters {
// base emitter
	vec2 position;
	float rotation;

// angular distribution
	float angleScalar;
	float cauchyMix;

// array modifier
	int repeats;
	float emitterSpacing;
	float width;
};

layout( set = 0, binding = 4 ) uniform emitterParameters {
	LightEmitterParameters emitterParams[ 256 ];
} EmitterParameters;

#define NOHIT						0
#define DIFFUSE						1
#define METALLIC					2
#define MIRROR						3

// air reserve value
#define AIR							5
// below this point, we have specific forms of glass
#define CAUCHY_FUSEDSILICA			6
#define CAUCHY_BOROSILICATE_BK7		7
#define CAUCHY_HARDCROWN_K5			8
#define CAUCHY_BARIUMCROWN_BaK4		9
#define CAUCHY_BARIUMFLINT_BaF10	10
#define CAUCHY_DENSEFLINT_SF10		11
// more coefficients available at https://web.archive.org/web/20151011033820/http://www.lacroixoptical.com/sites/default/files/content/LaCroix%20Dynamic%20Material%20Selection%20Data%20Tool%20vJanuary%202015.xlsm
#define SELLMEIER_BOROSILICATE_BK7	12
#define SELLMEIER_SAPPHIRE			13
#define SELLMEIER_FUSEDSILICA		14
#define SELLMEIER_MAGNESIUMFLOURIDE	15

struct intersectionResult {
// scene intersection representation etc loosely based on Daedalus
	float dist;
	float albedo;
	float IoR;
	float roughness;
	vec2 normal;
	bool frontFacing;
	int materialType;
};

intersectionResult getDefaultIntersection () {
	intersectionResult result;
	result.dist = 0.0f;
	result.albedo = 0.0f;
	result.IoR = 0.0f;
	result.roughness = 0.0f;
	result.normal = vec2( 0.0f );
	result.frontFacing = false;
	result.materialType = NOHIT;
	return result;
}

// for the values below that depend on access to the wavelength
float wavelength;

// global state tracking
int hitSurfaceType = 0;
float hitRoughness = 0.0f;
float hitAlbedo = 0.0f;

// raymarch parameters
const float epsilon = 0.03f;
const float maxDistance = 6000.0f;
const int maxSteps = 300;

// getting the wavelength-dependent IoR for materials
float evaluateCauchy ( float A, float B, float wms ) {
	return A + B / wms;
}

float evaluateSellmeier ( vec3 B, vec3 C, float wms ) {
	return sqrt( 1.0f + ( wms * B.x / ( wms - C.x ) ) + ( wms * B.y / ( wms - C.y ) ) + ( wms * B.z / ( wms - C.z ) ) );
}

// support for glass behavior
float Reflectance ( const float cosTheta, const float IoR ) {
	#if 0
	// Use Schlick's approximation for reflectance
	float r0 = ( 1.0f - IoR ) / ( 1.0f + IoR );
	r0 = r0 * r0;
	return r0 + ( 1.0f - r0 ) * pow( ( 1.0f - cosTheta ), 5.0f );
	#else
	// "Full Fresnel", from https://www.shadertoy.com/view/csfSz7
	float g = sqrt( IoR * IoR + cosTheta * cosTheta - 1.0f );
	float a = ( g - cosTheta ) / ( g + cosTheta );
	float b = ( ( g + cosTheta ) * cosTheta - 1.0f ) / ( ( g - cosTheta ) * cosTheta + 1.0f );
	return 0.5f * a * a * ( 1.0f + b * b );
	#endif
	//	another expression used here... https://www.shadertoy.com/view/wlyXzt - what's going on there?
}

float getIORForMaterial ( int material ) {
	// There are a couple ways to get IoR from wavelength
	float wavelengthMicrons = wavelength / 1000.0f;
	const float wms = wavelengthMicrons * wavelengthMicrons;

	float IoR = 0.0f;
	switch ( material ) {
		// Cauchy second order approx
		case CAUCHY_FUSEDSILICA:			IoR = evaluateCauchy( 1.4580f, 0.00354f, wms ); break;
		case CAUCHY_BOROSILICATE_BK7:		IoR = evaluateCauchy( 1.5046f, 0.00420f, wms ); break;
		case CAUCHY_HARDCROWN_K5:			IoR = evaluateCauchy( 1.5220f, 0.00459f, wms ); break;
		case CAUCHY_BARIUMCROWN_BaK4:		IoR = evaluateCauchy( 1.5690f, 0.00531f, wms ); break;
		case CAUCHY_BARIUMFLINT_BaF10:		IoR = evaluateCauchy( 1.6700f, 0.00743f, wms ); break;
		case CAUCHY_DENSEFLINT_SF10:		IoR = evaluateCauchy( 1.7280f, 0.01342f, wms ); break;
		// Sellmeier third order approx
		case SELLMEIER_BOROSILICATE_BK7:	IoR = evaluateSellmeier( vec3( 1.03961212f, 0.231792344f, 1.01046945f ), vec3( 1.01046945f, 6.00069867e-3f, 2.00179144e-2f ), wms ); break;
		case SELLMEIER_SAPPHIRE:			IoR = evaluateSellmeier( vec3( 1.43134930f, 0.650547130f, 5.34140210f ), vec3( 5.34140210f, 5.27992610e-3f, 1.42382647e-2f ), wms ); break;
		case SELLMEIER_FUSEDSILICA:			IoR = evaluateSellmeier( vec3( 0.69616630f, 0.407942600f, 0.89747940f ), vec3( 0.89747940f, 0.00467914800f, 0.01351206000f ), wms ); break;
		case SELLMEIER_MAGNESIUMFLOURIDE:	IoR = evaluateSellmeier( vec3( 0.48755108f, 0.398750310f, 2.31203530f ), vec3( 2.31203530f, 0.00188217800f, 0.00895188800f ), wms ); break;
		default: IoR = 1.0f;
	}

	return IoR;
}

bool isRefractive ( int id ) {
	return id >= CAUCHY_FUSEDSILICA;
}

mat2 Rotate2D ( in float a ) {
	float c = cos( a ), s = sin( a );
	return mat2( c, s, -s, c );
}

// Hash by David_Hoskins
#define UI0 1597334673U
#define UI1 3812015801U
#define UI2 uvec2(UI0, UI1)
#define UI3 uvec3(UI0, UI1, 2798796415U)
#define UIF (1.0 / float(0xffffffffU))

vec3 hash33( vec3 p ) {
	uvec3 q = uvec3( ivec3( p ) ) * UI3;
	q = ( q.x ^ q.y ^ q.z )*UI3;
	return -1.0 + 2.0 * vec3( q ) * UIF;
}

// should we invert the refractive stuff -> needs to be handled differently for lens elements?
bool invert = false;

float rectangle ( vec2 samplePosition, vec2 halfSize ) {
	vec2 componentWiseEdgeDistance = abs( samplePosition ) - halfSize;
	float outsideDistance = length( max( componentWiseEdgeDistance, 0 ) );
	float insideDistance = min( max( componentWiseEdgeDistance.x, componentWiseEdgeDistance.y ), 0 );
	return outsideDistance + insideDistance;
}

float sdParabola( in vec2 pos, in float wi, in float he ) {
	// "width" and "height" of a parabola segment
	pos.x = abs(pos.x);

	float ik = wi*wi/he;
	float p = ik*(he-pos.y-0.5*ik)/3.0;
	float q = pos.x*ik*ik*0.25;
	float h = q*q - p*p*p;

	float x;
	if( h>0.0 ) // 1 root
	{
		float r = sqrt(h);
		x = pow(q+r,1.0/3.0) + pow(abs(q-r),1.0/3.0)*sign(p);
	}
	else        // 3 roots
	{
		float r = sqrt(p);
		x = 2.0*r*cos(acos(q/(p*r))/3.0); // see https://www.shadertoy.com/view/WltSD7 for an implementation of cos(acos(x)/3) without trigonometrics
	}

	x = min(x,wi);

	return length(pos-vec2(x,he-x*x/ik)) *
	sign(ik*(pos.y-he)+pos.x*pos.x);
}

uvec3 murmurHash33(uvec3 src) {
	const uint M = 0x5bd1e995u;
	uvec3 h = uvec3(1190494759u, 2147483647u, 3559788179u);
	src *= M; src ^= src>>24u; src *= M;
	h *= M; h ^= src.x; h *= M; h ^= src.y; h *= M; h ^= src.z;
	h ^= h>>13u; h *= M; h ^= h>>15u;
	return h;
}

// 3 outputs, 3 inputs
vec3 hash(vec3 src) {
	uvec3 h = murmurHash33(floatBitsToUint(src));
	return uintBitsToFloat(h & 0x007fffffu | 0x3f800000u) - 1.0;
}

// returns offset to the center of the cell in the current tile
vec2 get_point(vec2 pos){
	vec2 p = floor(pos-1.)*3.;
	vec3 rng = hash(vec3(p, p.x+p.y))-.5;
	rng+=.5*vec3(cos(5.*rng.z), sin(5.*rng.z), .0);
	return rng.xy;
}

const vec2 offsets[24] = vec2[24](
vec2(-2, -2),
vec2(-1, -2),
vec2(0, -2),
vec2(1, -2),
vec2(2, -2),
vec2(-2, -1),
vec2(-1, -1),
vec2(0, -1),
vec2(1, -1),
vec2(2, -1),
vec2(-2, 0),
vec2(-1, 0),
vec2(1, 0),
vec2(2, 0),
vec2(-2, 1),
vec2(-1, 1),
vec2(0, 1),
vec2(1, 1),
vec2(2, 1),
vec2(-2, 2),
vec2(-1, 2),
vec2(0, 2),
vec2(1, 2),
vec2(2, 2)

);

float sdVoronoi(vec2 pos){
	pos-=.5;
	vec2 p = fract(pos)-.5;

	//nearest is initialized to the center cell because of potential future optimizations to reduce iteration count
	vec2 nearest = get_point(pos);
	vec2 lastnearest = vec2(100.);
	vec2 current = vec2(0.);

	vec3 squaredists = vec3(
	dot(p-nearest, p-nearest),
	dot(p-lastnearest, p-lastnearest),
	dot(p-current, p-current));

	//finding the two nearest points to the sampling point
	for (int i = 0; i<24; i++){
		current = offsets[i];
		current += get_point(pos+offsets[i]);
		squaredists.z = dot(p-current, p-current);
		if (squaredists.z<squaredists.y){
			squaredists.yz = squaredists.zy;
			lastnearest = current;
		}
		if (squaredists.y<squaredists.x){
			squaredists.xy = squaredists.yx;
			lastnearest = nearest;
			nearest = current;
		}
	}

	//float d = length(p-lastnearest)-.5+.5*(cos(iTime));
	float d = length(p-lastnearest)-.5;
	return d;
}

// for the walls
float rayPlaneIntersect ( in vec3 rayOrigin, in vec3 rayDirection ) {
	const vec3 normal = vec3( 0.0f, 1.0f, 0.0f );
	const vec3 planePt = vec3( 0.0f, 0.0f, 0.0f );
	return -( dot( rayOrigin - planePt, normal ) ) / dot( rayDirection, normal );
}

float de ( vec2 p ) {
	float sceneDist = 100000.0f;
	const vec2 pOriginal = p;

	hitAlbedo = 0.0f;
	hitSurfaceType = NOHIT;
	hitRoughness = 0.0f;

	{
		const float d = abs( sdParabola( p - vec2( GlobalData.floatBufferResolution.xy ) * vec2( 0.5f, 0.78f ), 1200.0f, 350.0f ) ) - 5.0f;
		sceneDist = min( sceneDist, d );
		if ( sceneDist == d && d < epsilon ) {
			hitSurfaceType = MIRROR;
			hitAlbedo = 1.0f;
		}
	}

//	{
//		const float scalar = 100.0f;
//		const float d = ( invert ? -1.0f : 1.0f ) * sdVoronoi( p / scalar ) * scalar;
//		sceneDist = min( sceneDist, d );
//		if ( sceneDist == d && d < epsilon ) {
//			hitSurfaceType = SELLMEIER_BOROSILICATE_BK7;
//			hitAlbedo = 1.0f;
//		}
//	}

	if ( true ) {
		p = Rotate2D( 0.1f ) * pOriginal;
		vec2 p2 = p;
		vec2 pBall = p - vec2( GlobalData.floatBufferResolution.xy / 2.0f );

		pModPolar( pBall, 12 );
		pBall = Rotate2D( 0.0f ) * pBall;
		pBall -= vec2( 200.0f, 0.0f );

		vec2 gridIndex;

		pModInterval1( p.y, 600.0f, -1.0f, 3.0f );

		gridIndex.x = pModInterval1( p.x, 64.0f, 1.0f, 20.0f );
		gridIndex.y = pModInterval1( p.y, 64.0f, 1.0f, 4.0f );
		vec2 gridIndex2;

		// adding a second order repeat
		pModInterval1( p2.y, 800.0f, -1.0f, 3.0f );

		gridIndex2.x = pModInterval1( p2.x, 6.0f, 20.0f, 600.0f );
		gridIndex2.y = pModInterval1( p2.y, 6.0f, 10.0f, 85.0f );
		{ // an example object (refractive)
			uint seedCache = seed;
			seed = 31415 * uint( gridIndex.x ) + uint( gridIndex.y ) * 42069 + 999999;
			const vec3 noise = 0.5f * hash33( vec3( gridIndex.xy, 0.0f ) ) + vec3( 2.0f, 1.0f, 0.5f );
			const vec3 noise2 = 0.5f * hash33( vec3( gridIndex2.xy, 0.0f ) ) + vec3( 2.0f, 1.0f, 0.5f );
			// const float d = ( invert ? -1.0f : 1.0f ) * ( ( noise.z > 0.25f ) ? ( distance( p, vec2( 0.0f ) ) - 2.0f * noise.x ) : ( ( distance( p, vec2( 0.0f ) ) - ( 4.0f * noise.y ) ) ) );
			const float d = ( invert ? -1.0f : 1.0f ) * ( min( distance( pBall, vec2( 300.0f, 0.0f ) ) - 100.0f,
				// min( ( noise.z > 0.25f ) ? ( distance( p, vec2( 0.0f ) ) - 10.0f * noise.z ) : ( rectangle( Rotate2D( 10.0f * noise.x ) * p, vec2( 2.40f * noise.y ) ) ),
				// ( noise2.z > 0.25f ) ? ( distance( p2, vec2( 0.0f ) ) - 1.0f * noise2.z ) : ( rectangle( Rotate2D( 100.0f * noise2.x ) * p2, vec2( 1.0f * noise2.y ) ) ) ) ) );
				min( ( distance( p, vec2( 0.0f ) ) - 28.0f ),
				( distance( p2, vec2( 0.0f ) ) - 2.0f ) ) ) );
			// const float d = ( invert ? -1.0f : 1.0f ) * ( distance( p, vec2( 0.0f ) ) - 28.0f );
//			 const float d = ( invert ? -1.0f : 1.0f ) * ( rectangle( Rotate2D( 100.0f * noise.z ) * p, vec2( 4.0f, 3.0f ) ) );
			seed = seedCache;
			sceneDist = min( sceneDist, d );
			if ( sceneDist == d && d < epsilon ) {
				hitSurfaceType = SELLMEIER_BOROSILICATE_BK7;
				hitAlbedo = 1.0f;
			}
		}
	}

	// walls at the edges of the screen for the rays to bounce off of
	if ( true ) {
		const float d = min( min( min(
		rectangle( pOriginal - vec2( 0.0f, 0.0f ), vec2( 4000.0f, 20.0f ) ),
		rectangle( pOriginal - vec2( 0.0f, GlobalData.floatBufferResolution.y ), vec2( 4000.0f, 20.0f ) ) ),
		rectangle( pOriginal - vec2( 0.0f, 0.0f ), vec2( 20.0f, 3000.0f ) ) ),
		rectangle( pOriginal - vec2( GlobalData.floatBufferResolution.x, 0.0f ), vec2( 20.0f, 3000.0f ) ) );
		sceneDist = min( sceneDist, d );
		if ( sceneDist == d && d < epsilon ) {
			hitSurfaceType = DIFFUSE;
			hitAlbedo = 0.9f;
		}
	}

	// get back final result
	return sceneDist;
}

// function to get the normal
vec2 SDFNormal ( vec2 p ) {
	const vec2 k = vec2( 1.0f, -1.0f );
	return normalize(
		k.xx * de( p + k.xx * epsilon ).x +
		k.xy * de( p + k.xy * epsilon ).x +
		k.yx * de( p + k.yx * epsilon ).x +
		k.yy * de( p + k.yy * epsilon ).x );
}

// trace against the scene
intersectionResult sceneTrace ( vec2 rayOrigin, vec2 rayDirection ) {
	intersectionResult result = getDefaultIntersection();

	// is the initial sample point inside? -> toggle invert so we correctly handle refractive objects
	if ( de( rayOrigin ) < 0.0f ) { // this is probably a solution for the same problem in Daedalus, too...
		invert = !invert;
	}

	// if, after managing potential inversion, we still get a negative result back... we are inside solid scene geometry
	if ( de( rayOrigin ) < 0.0f ) {
		result.dist = -1.0f;
		result.materialType = NOHIT;
		result.albedo = hitAlbedo;
	} else {
		// we're in a valid location and clear to do a raymarch
		result.dist = 0.0f;
		for ( int i = 0; i < maxSteps; i++ ) {
			float d = de( rayOrigin + result.dist * rayDirection );
			if ( d < epsilon ) {
				// we have a hit - gather intersection information
				result.materialType = hitSurfaceType;
				result.albedo = hitAlbedo;
				result.frontFacing = !invert; // for now, this will be sufficient to make decisions re: IoR
				result.IoR = getIORForMaterial( hitSurfaceType );
				result.normal = SDFNormal( rayOrigin + result.dist * rayDirection );
				result.roughness = hitRoughness;
			} else if ( result.dist > maxDistance ) {
				result.materialType = NOHIT;
				break;
			}
			result.dist += d;
		}
	}

	// and give back whatever we got
	return result;
}

void main () {
	// pixel index
	uint loc = uint( gl_GlobalInvocationID.x );
	uint baseIdx = loc * GlobalData.numBounces;

	// seeding RNG, unique per invocation
	seed = PushConstants.wangSeed + 8675309 * loc.x;

	// the raytrace process...
	vec2 rayOrigin, rayDirection;


	// picking a light...
	uint lightPick = getPickedLight();
	LightEmitterParameters params = EmitterParameters.emitterParams[ lightPick ];

	// cache rotation matrix
	const mat2 rot = Rotate2D( params.rotation );
	const vec2 subpixelJitter = vec2( NormalizedRandomFloat(), NormalizedRandomFloat() );

	// values in the buffer set origin, direction
	float pickedRepeat = 0;
	if ( params.repeats != 1 ) {
		pickedRepeat = float( floor( NormalizedRandomFloat() * params.repeats ) ) - float( params.repeats ) / 2.0f;
	}
	vec2 offset = rot * pickedRepeat * params.emitterSpacing * vec2( 1.0f, 0.0f );

	if ( lightPick == 0 ) {
		// this is the mouse light
		rayOrigin = subpixelJitter + GlobalData.mouseLoc + offset + params.width * rot * vec2( NormalizedRandomFloat() - 0.5f, 0.0f );
	} else {
		rayOrigin = subpixelJitter + params.position + offset + params.width * rot * vec2( NormalizedRandomFloat() - 0.5f, 0.0f );
	}
	// direction is the same either way
	rayDirection = normalize( Rotate2D( params.rotation + params.angleScalar * ( NormalizedRandomFloat() - 0.5f ) + params.cauchyMix * rnd_disc_cauchy().x ) * vec2( 0.0f, 1.0f ) );

	// picking a wavelength...
		// importance sampled from the light
	wavelength = getWavelengthForLight( lightPick );

	// initial values... probably redundant
	float transmission = 1.0f;
	float energy = 1.0f;

	bool deadRay = false;
	for ( int i = 0; i < GlobalData.numBounces; i++ ) {
		// we only draw segments until the ray "dies"
		if ( !deadRay ) {

			// do the scene intersection
			intersectionResult result = sceneTrace( rayOrigin, rayDirection );

			// add the line to the system
			raySegment r = getDefaultSegment();
			r.a = rayOrigin;
			r.a.x = remap( r.a.x, 0.0f, GlobalData.floatBufferResolution.x, -1.0f, 1.0f );
			r.a.y = remap( r.a.y, 0.0f, GlobalData.floatBufferResolution.y, -1.0f, 1.0f );

			r.b = rayOrigin + result.dist * rayDirection;
			r.b.x = remap( r.b.x, 0.0f, GlobalData.floatBufferResolution.x, -1.0f, 1.0f );
			r.b.y = remap( r.b.y, 0.0f, GlobalData.floatBufferResolution.y, -1.0f, 1.0f );

			r.brightness = energy;
			r.wavelength = wavelength;
			rays[ baseIdx + i ] = r;

			// evaluating the russian roulette termination...
			if ( NormalizedRandomFloat() > energy )
				deadRay = true;
			energy *= 1.0f / min( energy, 1.0f ); // compensation term

			if ( energy < 0.001f ) deadRay = true;

			// evaluating the albedo's effect on transmission + energy
			transmission *= result.albedo;
			energy *= result.albedo;

			// epsilon bump + update origin
			rayOrigin = rayOrigin + result.dist * rayDirection + result.normal * epsilon * 3.0f;

			// switch on material type
			switch ( result.materialType ) {
			case DIFFUSE:
				rayDirection = normalize( CircleOffset() );
				// invert if going into the surface
				if ( dot( rayDirection, result.normal ) < 0.0f ) {
					rayDirection = -rayDirection;
				}
				break;

			case METALLIC:
				// todo
				break;

			case MIRROR:
				rayDirection = reflect( rayDirection, result.normal );
				break;

				// below this point, we have to consider the IoR for the specific form of glass... because we precomputed all the
				// varying behavior already, we can just treat it uniformly, only need to consider frontface/backface for inversion
			default:
				rayOrigin -= result.normal * epsilon * 5;
				result.IoR = result.frontFacing ? ( 1.0f / result.IoR ) : ( result.IoR ); // "reverse" back to physical properties for IoR
				float cosTheta = min( dot( -normalize( rayDirection ), result.normal ), 1.0f );
				float sinTheta = sqrt( 1.0f - cosTheta * cosTheta );
				bool cannotRefract = ( result.IoR * sinTheta ) > 1.0f; // accounting for TIR effects
				if ( cannotRefract || Reflectance( cosTheta, result.IoR ) > NormalizedRandomFloat() ) {
					rayDirection = normalize( mix( reflect( normalize( rayDirection ), result.normal ), CircleOffset(), result.roughness ).xy );
				} else {
					rayDirection = normalize( mix( refract( normalize( rayDirection ), result.normal, result.IoR ), CircleOffset(), result.roughness ).xy );
				}
				break;
			}
		} else {
			// if the ray has finished tracing, we need to zero out the rest of the segment memory, so the raster process doesn't draw anything
			rays[ baseIdx + i ] = getDefaultSegment();
		}
	}
}