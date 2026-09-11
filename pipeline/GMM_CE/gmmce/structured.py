"""
DFT / circulant-embedding machinery used by the structured-covariance GMM
variants (Sec. IV-A, IV-B of the paper).

- `dft_matrix(M)`: unitary M x M DFT matrix.
- `toeplitz_embedding_matrix(M)`: Q_M, the 2M x M matrix formed by the
  first M columns of the unitary 2M x 2M DFT matrix (paper, Sec. IV-A,
  citing Strang 1986). Q_M has orthonormal columns (Q_M^H Q_M = I_M) and
  Q_M^H diag(c) Q_M is Hermitian Toeplitz for any c in R^{2M}_+
  (a principal submatrix of a circulant matrix is Toeplitz).
- `block_operator(dims, kind)`: Kronecker product of per-dimension
  operators (DFT or Toeplitz-embedding), in the order matching the
  vec() convention of `gmmce.pilots` (first dim = outer/slow, last =
  inner/fast). For the 2D case this is Q~ = Q_{Nt} (x) Q_{Nc}.
"""
from __future__ import annotations

from functools import reduce

import numpy as np


def dft_matrix(M: int) -> np.ndarray:
    j = np.arange(M)
    return np.exp(-2j * np.pi * np.outer(j, j) / M) / np.sqrt(M)


def toeplitz_embedding_matrix(M: int) -> np.ndarray: # toeplize 행렬 대각화시키기 위해 필요. 본 논문에서 Q_M의미.  
    F2M = dft_matrix(2 * M)
    return F2M[:, :M]


def block_operator(dims: list[int], kind: str) -> np.ndarray: # dims = [Nt, Nc]이런식으로 입력받고, dft or toeplitz중 하나의 종류 입력받아서 최종 diagonalized form에서 F or Q~ 반환 
    """kind: 'dft' -> kron of unitary DFT matrices (circulant case)
             'toeplitz' -> kron of Q_M matrices (Toeplitz case)
    dims ordered outer(slow) -> inner(fast), e.g. [Nt, Nc]."""
    if kind == 'dft':
        mats = [dft_matrix(M) for M in dims]
    elif kind == 'toeplitz':
        mats = [toeplitz_embedding_matrix(M) for M in dims]
    else:
        raise ValueError(kind)
    return reduce(np.kron, mats) # np.kron 크로네커곱 연산을 수행함. 순서바꾸기 불가. reduce(np.kron, mats)는 mats에 있는 여러 행렬을 좌에서부터 우로 2개씩 순차적으로 크로네커곱해서 하나의 matrix로 줄이는 명령어.


def _self_check_kron_convention(): # 크로네커곱 연산 순서가 잘 맞는지 체크하는 함수
    """Verify kron(C_time, C_freq) @ vec(H) == vec(C_freq @ H @ C_time.T)."""
    from .pilots import vec
    rng = np.random.default_rng(0)
    Nc, Nt = 5, 3                                                       # 임의데이터 생성
    H = rng.normal(size=(Nc, Nt)) + 1j * rng.normal(size=(Nc, Nt))      # 임의데이터 생성
    C_time = rng.normal(size=(Nt, Nt)) + 1j * rng.normal(size=(Nt, Nt)) # 임의데이터 생성
    C_freq = rng.normal(size=(Nc, Nc)) + 1j * rng.normal(size=(Nc, Nc)) # 임의데이터 생성
    lhs = np.kron(C_time, C_freq) @ vec(H)
    rhs = vec(C_freq @ H @ C_time.T)
    assert np.allclose(lhs, rhs), "kron/vec convention mismatch"  # 컴퓨터의 소수 연산(부동소수점)은 정밀도 한계 때문에 그냥 ==로 비교를 하면 실패하는 경우가 많음. 
                                                                  # 정해진 오차 범위(기본 허용 오차: relative tolerance $10^{-5}$, absolute tolerance $10^{-8}$) 내에서 동일한지 확인


def _self_check_toeplitz_projection(): # toeplize 특성 점검 
    rng = np.random.default_rng(0)
    M = 6
    Q = toeplitz_embedding_matrix(M)
    assert np.allclose(Q.conj().T @ Q, np.eye(M), atol=1e-10), "Q_M columns not orthonormal" # 직교성 검사
    c = rng.uniform(0.1, 1.0, size=2 * M)
    C = Q.conj().T @ np.diag(c) @ Q
    assert np.allclose(C, C.conj().T, atol=1e-10), "not Hermitian" # 전치가 같은지
    # Toeplitz check: constant along diagonals
    for k in range(1, M):
        diag_vals = np.diag(C, k)
        assert np.allclose(diag_vals, diag_vals[0], atol=1e-8), f"not Toeplitz at offset {k}" # 대각 원소가 같은지 검사


if __name__ == "__main__":
    _self_check_kron_convention()
    _self_check_toeplitz_projection()
    print("structured.py self-checks passed")
