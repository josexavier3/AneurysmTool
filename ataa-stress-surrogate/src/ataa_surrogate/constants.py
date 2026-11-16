"""Canonical dimensions, component order and unit conversion for the ATAA surrogate."""

N_NODES = 1354
N_COMPONENTS = 6
IN_DIM = 10
FIELD_WIDTH = N_NODES * N_COMPONENTS
OUT_DIM = 2 * FIELD_WIDTH
# VTK 6-component stress and the rebuild's default unpacking of S.
COMPONENT_ORDER = ("11", "22", "33", "12", "23", "13")
# preProcessing.py stores C = F.T@F then takes row-major entries [0,4,8,1,3,2],
# i.e. C11,C22,C33,C12,C21,C13. Slot 4 is C21, not C23.
C_SOURCE_FLAT9_INDICES = (0, 4, 8, 1, 3, 2)
C_SOURCE_COMPONENT_ORDER = ("11", "22", "33", "12", "21", "13")
DYNE_CM2_TO_KPA = 1e-4
