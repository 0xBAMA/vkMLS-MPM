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
	point p = points[ idx ];

	// points make atomic writes on the buffers
		// ( momentum, mass )

	if ( p.particleType == 0 ) { // hookean

		mat2 stress = mat2( 0.0f );
		mat2 F = p.Fs;

		float J = determinant( F );

		// MPM course, page 46
		float volume = p.v0 * J;

		// useful matrices for Neo-Hookean model
		mat2 F_T = transpose( F );
		mat2 F_inv_T = inverse( F_T );
		mat2 F_minus_F_inv_T = F - F_inv_T;

		// MPM course equation 48
		mat2 P_term_0 = GlobalData.elasticMu * ( F_minus_F_inv_T );
		mat2 P_term_1 = GlobalData.elasticLambda * log( J ) * F_inv_T;
		mat2 P = P_term_0 + P_term_1;

		// cauchy_stress = (1 / det(F)) * P * F_T
		// equation 38, MPM course
		stress = ( 1.0f / J ) * ( P * F_T );

		// (M_p)^-1 = 4, see APIC paper and MPM course page 42
		// this term is used in MLS-MPM paper eq. 16. with quadratic weights, Mp = (1/4) * (delta_x)^2.
		// in this simulation, delta_x = 1, because i scale the rendering of the domain rather than the domain itself.
		// we multiply by dt as part of the process of fusing the momentum and force update for MLS-MPM
		mat2 eq_16_term_0 = -volume * 4 * stress * GlobalData.dT;

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
				vec2 cellDist = ( vec2( cellIdxInner) - p.position ) + 0.5f;
				vec2 Q = p.C * cellDist;

				// MPM course, equation 172
				float weightedMass = weight * p.mass;
				imageAtomicAdd( massAtomic, cellIdxInner, int( weightedMass * GlobalData.fixedPointScalar ) );

				// velocity grid contribution...
				vec2 writeV = vec2( 0.0f );

				// APIC P2G momentum contribution
				writeV += weightedMass * ( p.velocity + Q );

				// fused force/momentum update from MLS-MPM
				// see MLS-MPM paper, equation listed after eqn. 28
				vec2 momentum = ( eq_16_term_0 * weight ) * cellDist;
				writeV += momentum;

				// fixed point adjustment applied on write and in reverse on read
				imageAtomicAdd( velocityXAtomic, cellIdxInner, int( writeV.x * GlobalData.fixedPointScalar ) );
				imageAtomicAdd( velocityYAtomic, cellIdxInner, int( writeV.y * GlobalData.fixedPointScalar ) );

				// total update on cell.v is now:
				// weight * (dt * M^-1 * p.volume * p.stress + p.mass * p.C)
				// this is the fused momentum + force from MLS-MPM. however, instead of our stress being derived from the energy density,
				// i use the weak form with cauchy stress. converted:
				// p.volume_0 * (dΨ/dF)(Fp)*(Fp_transposed)
				// is equal to p.volume * σ

				// note: currently "cell.v" refers to MOMENTUM, not velocity!
				// this gets converted in the UpdateGrid step below.

			}
		}
	}
}