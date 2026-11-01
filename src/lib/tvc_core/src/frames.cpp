#include "tvc/frames.hpp"

namespace tvc {

// R_ned←enu = [[0,1,0],[1,0,0],[0,0,-1]]
// R_frd←flu = [[1,0,0],[0,-1,0],[0,0,-1]]
// R_ned←frd = R_ned←enu * R_enu←flu * R_flu←frd
// with R_flu←frd = R_frd←flu^T = R_frd←flu (involutory)

static void mat3_mul(const float A[9], const float B[9], float C[9])
{
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 3; c++) {
			C[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c]
				       + A[r * 3 + 1] * B[1 * 3 + c]
				       + A[r * 3 + 2] * B[2 * 3 + c];
		}
	}
}

static void mat3_transpose(const float A[9], float AT[9])
{
	AT[0] = A[0]; AT[1] = A[3]; AT[2] = A[6];
	AT[3] = A[1]; AT[4] = A[4]; AT[5] = A[7];
	AT[6] = A[2]; AT[7] = A[5]; AT[8] = A[8];
}

Quat q_sim_to_px4(const Quat &q_sim)
{
	// R_enu←flu from q_sim
	float R_enu_flu[9];
	q_to_dcm(q_sim, R_enu_flu);

	const float R_ned_enu[9] = {
		0.f, 1.f, 0.f,
		1.f, 0.f, 0.f,
		0.f, 0.f, -1.f
	};
	const float R_frd_flu[9] = {
		1.f, 0.f, 0.f,
		0.f, -1.f, 0.f,
		0.f, 0.f, -1.f
	};
	float R_flu_frd[9];
	mat3_transpose(R_frd_flu, R_flu_frd);

	float tmp[9];
	float R_ned_frd[9];
	mat3_mul(R_enu_flu, R_flu_frd, tmp);
	mat3_mul(R_ned_enu, tmp, R_ned_frd);
	return q_from_dcm(R_ned_frd);
}

Quat q_px4_to_sim(const Quat &q_px4)
{
	// Inverse: R_enu←flu = R_ned←enu^T * R_ned←frd * R_frd←flu
	float R_ned_frd[9];
	q_to_dcm(q_px4, R_ned_frd);

	const float R_ned_enu[9] = {
		0.f, 1.f, 0.f,
		1.f, 0.f, 0.f,
		0.f, 0.f, -1.f
	};
	// involutory
	const float R_frd_flu[9] = {
		1.f, 0.f, 0.f,
		0.f, -1.f, 0.f,
		0.f, 0.f, -1.f
	};

	float tmp[9];
	float R_enu_flu[9];
	mat3_mul(R_ned_frd, R_frd_flu, tmp);
	mat3_mul(R_ned_enu, tmp, R_enu_flu); // R_ned_enu == R_ned_enu^T
	return q_from_dcm(R_enu_flu);
}

} // namespace tvc
