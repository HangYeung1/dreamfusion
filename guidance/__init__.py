from typing import Union

from .IFGuide import IFGuide
from .StableGuide import StableGuide

GuideDict = {"StableGuide": StableGuide, "IFGuide": IFGuide}
GuideList = GuideDict.keys()
GuideTypes = Union[*GuideDict.values()]