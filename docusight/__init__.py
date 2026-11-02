"""DocuSight AI - layout-aware document intelligence."""

__version__ = "0.1.0"

# On Windows, PaddlePaddle and PyTorch ship conflicting DLLs; loading torch first
# avoids "WinError 127 ... shm.dll" when paddleocr later imports torch indirectly.
try:  # pragma: no cover - environment specific
    import torch  # noqa: F401
except Exception:
    pass

DOC_TYPES = ("bank_statement", "invoice", "payslip", "kyc_id")
