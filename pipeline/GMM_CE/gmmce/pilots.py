"""
Pilot allocation (lattice-type, Fig. 1 of the paper) and vec()/unvec()
conventions used throughout this package.

Vectorization convention
-------------------------
H has shape (Nc, Nt) (subcarrier x time symbol). We vectorize it
column-major ("Fortran order"), i.e.

    h = vec(H) = H.flatten(order='F')   # shape (Nc*Nt,)

so that the subcarrier index varies fastest and the time-symbol index
varies slowest: h[t*Nc + c] = H[c, t]. With this convention, a separable
(Kronecker) covariance is

    C = kron(C_time, C_freq)   # C_time: (Nt,Nt), C_freq: (Nc,Nc)

which matches the paper's Q~ = Q_{Nt} (x) Q_{Nc} ordering (time is the
"outer"/slow factor, frequency the "inner"/fast factor). See
`gmmce.structured` for a numerical check of this identity.
"""
from __future__ import annotations # Type Hint의 평가 연기 3.14면 없어도 됨.

import numpy as np


def vec(H: np.ndarray) -> np.ndarray:
    """H: (..., Nc, Nt) -> h: (..., Nc*Nt), column-major (freq fastest)."""
    *batch, Nc, Nt = H.shape  # batch에 맨 끝 행, 열 정보 빼고 다 묶어 리스트화해서 batch에 저장
    return np.reshape(np.swapaxes(H, -1, -2), (*batch, Nc * Nt)) # batch차원은 유지하고, Nc * Nt길이로 펴는 과정. python에서 reshape은 row-wise라 column wise로 바꿔주려고 축 순서 변환


def unvec(h: np.ndarray, Nc: int, Nt: int) -> np.ndarray:
    """Inverse of vec(): h: (..., Nc*Nt) -> H: (..., Nc, Nt)."""
    *batch, _ = h.shape
    H_T = np.reshape(h, (*batch, Nt, Nc))
    return np.swapaxes(H_T, -1, -2)


def lattice_pilot_indices(Nc: int, Nt: int, Npc: int, Npt: int) -> tuple[np.ndarray, np.ndarray]: # -> 는 return값 유형.
    """Evenly-spaced rectangular pilot lattice (Fig. 1: 'lattice-type').

    Returns (pilot_carriers, pilot_symbols), each a sorted 1D int array.
    Np = Npc * Npt pilots sit at every combination of these indices, i.e.
    a genuine Cartesian-product ("separable") pilot grid, which is what
    the Kronecker / 2x1D estimators require (A = A_t (x) A_c).
    """
    pilot_carriers = np.unique(np.round(np.linspace(0, Nc - 1, Npc)).astype(int)) #astype(int)는 정수형으로 반환, np.unique 는 배열의 중복인덱스 제거하고 오름차순으로 정렬
    pilot_symbols = np.unique(np.round(np.linspace(0, Nt - 1, Npt)).astype(int))
    if len(pilot_carriers) != Npc or len(pilot_symbols) != Npt:
        raise ValueError("Requested pilot counts do not fit evenly into (Nc, Nt)")
    return pilot_carriers, pilot_symbols # 파일럿이 들어가는 time, freq 인덱스 위치 반환


def selection_matrix(Nc: int, Nt: int, pilot_carriers: np.ndarray, pilot_symbols: np.ndarray) -> np.ndarray:
    """Build the 0/1 selection matrix A such that y = A @ vec(H).

    Rows of A are ordered as symbol-major, i.e. for t in pilot_symbols,
    for c in pilot_carriers: pick H[c, t]. This matches how observations
    are naturally read off in the 2x1D cascade.
    """
    D = Nc * Nt
    Np = len(pilot_carriers) * len(pilot_symbols)
    A = np.zeros((Np, D), dtype=np.float64)
    row = 0
    for t in pilot_symbols:
        for c in pilot_carriers:
            col = t * Nc + c  # matches vec() convention: index = t*Nc + c
            A[row, col] = 1.0
            row += 1
    return A


class PilotGrid:
    """Convenience container bundling pilot geometry + selection matrices."""

    def __init__(self, Nc: int, Nt: int, Npc: int, Npt: int): # self.붙은 애들은 객체가 삭제될 때까지 내부 메모리에 남아 지속되는 객체 고유 변수
        self.Nc, self.Nt = Nc, Nt
        self.Npc, self.Npt = Npc, Npt
        self.pilot_carriers, self.pilot_symbols = lattice_pilot_indices(Nc, Nt, Npc, Npt)
        self.A = selection_matrix(Nc, Nt, self.pilot_carriers, self.pilot_symbols)
        self.Np = self.A.shape[0]

    def observe(self, H: np.ndarray) -> np.ndarray:
        """H: (..., Nc, Nt) -> y: (..., Np) noiseless pilot observations."""
        h = vec(H) # row vector가 배치수 만큼 쌓인 형태가 python 기본형.
        return h @ self.A.T # 파이썬에서 행렬곱은 "@"", 
