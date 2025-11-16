import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Cannot be imported directly because `_typeshed` does not exist in the runtime
    from _typeshed import StrPath
else:
    type StrPath = str | os.PathLike[str]
