import copy

import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, GUID_WICPixelFormat32bppBGR, WICDecoder, GUID_WICPixelFormat32bppBGRA, \
    GUID_WICPixelFormat32bppPBGRA

from pyglet import image
import ctypes
import math
from pyglet import com
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from ctypes import *
import os
import platform

try:
    dwrite = 'dwrite'

    # System32 and SysWOW64 folders are opposite perception in Windows x64.
    # System32 = x64 dll's | SysWOW64 = x86 dlls
    # By default ctypes only seems to look in system32 regardless of Python architecture, which has x64 dlls.
    if platform.architecture()[0] == '32bit':
        if platform.machine().endswith('64'):  # Machine is 64 bit, Python is 32 bit.
            dwrite = os.path.join(os.environ['WINDIR'], 'SysWOW64', 'dwrite.dll')

    dwrite_lib = ctypes.windll.LoadLibrary(dwrite)
except OSError as err:
    # Doesn't exist? Should stop import of library.
    pass


def DWRITE_MAKE_OPENTYPE_TAG(a, b, c, d):
    return ord(d) << 24 | ord(c) << 16 | ord(b) << 8 | ord(a)


DWRITE_FACTORY_TYPE = UINT
DWRITE_FACTORY_TYPE_SHARED = 0
DWRITE_FACTORY_TYPE_ISOLATED = 1

DWRITE_FONT_WEIGHT = UINT
DWRITE_FONT_WEIGHT_THIN = 100
DWRITE_FONT_WEIGHT_EXTRA_LIGHT = 200
DWRITE_FONT_WEIGHT_ULTRA_LIGHT = 200
DWRITE_FONT_WEIGHT_LIGHT = 300
DWRITE_FONT_WEIGHT_SEMI_LIGHT = 350
DWRITE_FONT_WEIGHT_NORMAL = 400
DWRITE_FONT_WEIGHT_REGULAR = 400
DWRITE_FONT_WEIGHT_MEDIUM = 500
DWRITE_FONT_WEIGHT_DEMI_BOLD = 600
DWRITE_FONT_WEIGHT_SEMI_BOLD = 600
DWRITE_FONT_WEIGHT_BOLD = 700
DWRITE_FONT_WEIGHT_EXTRA_BOLD = 800
DWRITE_FONT_WEIGHT_ULTRA_BOLD = 800
DWRITE_FONT_WEIGHT_BLACK = 900
DWRITE_FONT_WEIGHT_HEAVY = 900
DWRITE_FONT_WEIGHT_EXTRA_BLACK = 950

name_to_weight = {"thin": DWRITE_FONT_WEIGHT_THIN,
                  "extralight": DWRITE_FONT_WEIGHT_EXTRA_LIGHT,
                  "ultralight": DWRITE_FONT_WEIGHT_ULTRA_LIGHT,
                  "light": DWRITE_FONT_WEIGHT_LIGHT,
                  "semilight": DWRITE_FONT_WEIGHT_SEMI_LIGHT,
                  "normal": DWRITE_FONT_WEIGHT_NORMAL,
                  "regular": DWRITE_FONT_WEIGHT_REGULAR,
                  "medium": DWRITE_FONT_WEIGHT_MEDIUM,
                  "demibold": DWRITE_FONT_WEIGHT_DEMI_BOLD,
                  "semibold": DWRITE_FONT_WEIGHT_SEMI_BOLD,
                  "bold": DWRITE_FONT_WEIGHT_BOLD,
                  "extrabold": DWRITE_FONT_WEIGHT_EXTRA_BOLD,
                  "ultrabold": DWRITE_FONT_WEIGHT_ULTRA_BOLD,
                  "black": DWRITE_FONT_WEIGHT_BLACK,
                  "heavy": DWRITE_FONT_WEIGHT_HEAVY,
                  "extrablack": DWRITE_FONT_WEIGHT_EXTRA_BLACK,
                  }

DWRITE_FONT_STRETCH = UINT
DWRITE_FONT_STRETCH_UNDEFINED = 0
DWRITE_FONT_STRETCH_ULTRA_CONDENSED = 1
DWRITE_FONT_STRETCH_EXTRA_CONDENSED = 2
DWRITE_FONT_STRETCH_CONDENSED = 3
DWRITE_FONT_STRETCH_SEMI_CONDENSED = 4
DWRITE_FONT_STRETCH_NORMAL = 5
DWRITE_FONT_STRETCH_MEDIUM = 5
DWRITE_FONT_STRETCH_SEMI_EXPANDED = 6
DWRITE_FONT_STRETCH_EXPANDED = 7
DWRITE_FONT_STRETCH_EXTRA_EXPANDED = 8

name_to_stretch = {"undefined": DWRITE_FONT_STRETCH_UNDEFINED,
                   "ultracondensed": DWRITE_FONT_STRETCH_ULTRA_CONDENSED,
                   "extracondensed": DWRITE_FONT_STRETCH_EXTRA_CONDENSED,
                   "condensed": DWRITE_FONT_STRETCH_CONDENSED,
                   "semicondensed": DWRITE_FONT_STRETCH_SEMI_CONDENSED,
                   "normal": DWRITE_FONT_STRETCH_NORMAL,
                   "medium": DWRITE_FONT_STRETCH_MEDIUM,
                   "semiexpanded": DWRITE_FONT_STRETCH_SEMI_EXPANDED,
                   "expanded": DWRITE_FONT_STRETCH_EXPANDED,
                   "extraexpanded": DWRITE_FONT_STRETCH_EXTRA_EXPANDED,
                   }

DWRITE_FONT_STYLE = UINT
DWRITE_FONT_STYLE_NORMAL = 0
DWRITE_FONT_STYLE_OBLIQUE = 1
DWRITE_FONT_STYLE_ITALIC = 2

name_to_style = {"normal": DWRITE_FONT_STYLE_NORMAL,
                 "oblique": DWRITE_FONT_STYLE_OBLIQUE,
                 "italic": DWRITE_FONT_STYLE_ITALIC}

UINT8 = c_uint8
UINT16 = c_uint16
INT16 = c_int16
INT32 = c_int32
UINT32 = c_uint32
UINT64 = c_uint64


class DWRITE_TEXT_METRICS(ctypes.Structure):
    _fields_ = (
        ('left', FLOAT),
        ('top', FLOAT),
        ('width', FLOAT),
        ('widthIncludingTrailingWhitespace', FLOAT),
        ('height', FLOAT),
        ('layoutWidth', FLOAT),
        ('layoutHeight', FLOAT),
        ('maxBidiReorderingDepth', UINT32),
        ('lineCount', UINT32),
    )


class DWRITE_FONT_METRICS(ctypes.Structure):
    _fields_ = (
        ('designUnitsPerEm', UINT16),
        ('ascent', UINT16),
        ('descent', UINT16),
        ('lineGap', INT16),
        ('capHeight', UINT16),
        ('xHeight', UINT16),
        ('underlinePosition', INT16),
        ('underlineThickness', UINT16),
        ('strikethroughPosition', INT16),
        ('strikethroughThickness', UINT16),
    )


class DWRITE_GLYPH_METRICS(ctypes.Structure):
    _fields_ = (
        ('leftSideBearing', INT32),
        ('advanceWidth', UINT32),
        ('rightSideBearing', INT32),
        ('topSideBearing', INT32),
        ('advanceHeight', UINT32),
        ('bottomSideBearing', INT32),
        ('verticalOriginY', INT32),
    )


class DWRITE_GLYPH_OFFSET(ctypes.Structure):
    _fields_ = (
        ('advanceOffset', FLOAT),
        ('ascenderOffset', FLOAT),
    )

    def __repr__(self):
        return f"DWRITE_GLYPH_OFFSET({self.advanceOffset}, {self.ascenderOffset})"


class DWRITE_CLUSTER_METRICS(ctypes.Structure):
    _fields_ = (
        ('width', FLOAT),
        ('length', UINT16),
        ('canWrapLineAfter', UINT16, 1),
        ('isWhitespace', UINT16, 1),
        ('isNewline', UINT16, 1),
        ('isSoftHyphen', UINT16, 1),
        ('isRightToLeft', UINT16, 1),
        ('padding', UINT16, 11),
    )


class IDWriteFontFace(com.pIUnknown):
    _methods_ = [
        ('GetType',
         com.STDMETHOD()),
        ('GetFiles',
         com.STDMETHOD()),
        ('GetIndex',
         com.STDMETHOD()),
        ('GetSimulations',
         com.STDMETHOD()),
        ('IsSymbolFont',
         com.STDMETHOD()),
        ('GetMetrics',
         com.METHOD(c_void, POINTER(DWRITE_FONT_METRICS))),
        ('GetGlyphCount',
         com.STDMETHOD()),
        ('GetDesignGlyphMetrics',
         com.STDMETHOD(POINTER(UINT16), UINT32, POINTER(DWRITE_GLYPH_METRICS), BOOL)),
        ('GetGlyphIndices',
         com.STDMETHOD(POINTER(UINT32), UINT32, POINTER(UINT16))),
        ('TryGetFontTable',
         com.STDMETHOD(UINT32, c_void_p, POINTER(UINT32), c_void_p, POINTER(BOOL))),
        ('ReleaseFontTable',
         com.METHOD(c_void)),
        ('GetGlyphRunOutline',
         com.STDMETHOD()),
        ('GetRecommendedRenderingMode',
         com.STDMETHOD()),
        ('GetGdiCompatibleMetrics',
         com.STDMETHOD()),
        ('GetGdiCompatibleGlyphMetrics',
         com.STDMETHOD()),
    ]


IID_IDWriteFontFace1 = com.GUID(0xa71efdb4, 0x9fdb, 0x4838, 0xad, 0x90, 0xcf, 0xc3, 0xbe, 0x8c, 0x3d, 0xaf)


class IDWriteFontFace1(IDWriteFontFace, com.pIUnknown):
    _methods_ = [
        ('GetMetric1',
         com.STDMETHOD()),
        ('GetGdiCompatibleMetrics1',
         com.STDMETHOD()),
        ('GetCaretMetrics',
         com.STDMETHOD()),
        ('GetUnicodeRanges',
         com.STDMETHOD()),
        ('IsMonospacedFont',
         com.STDMETHOD()),
        ('GetDesignGlyphAdvances',
         com.METHOD(c_void, POINTER(DWRITE_FONT_METRICS))),
        ('GetGdiCompatibleGlyphAdvances',
         com.STDMETHOD()),
        ('GetKerningPairAdjustments',
         com.STDMETHOD(UINT32, POINTER(UINT16), POINTER(INT32))),
        ('HasKerningPairs',
         com.METHOD(BOOL)),
        ('GetRecommendedRenderingMode1',
         com.STDMETHOD()),
        ('GetVerticalGlyphVariants',
         com.STDMETHOD()),
        ('HasVerticalGlyphVariants',
         com.STDMETHOD())
    ]


DWRITE_SCRIPT_SHAPES = UINT
DWRITE_SCRIPT_SHAPES_DEFAULT = 0


class DWRITE_SCRIPT_ANALYSIS(ctypes.Structure):
    _fields_ = (
        ('script', UINT16),
        ('shapes', DWRITE_SCRIPT_SHAPES),
    )


DWRITE_FONT_FEATURE_TAG = UINT


class DWRITE_FONT_FEATURE(ctypes.Structure):
    _fields_ = (
        ('nameTag', DWRITE_FONT_FEATURE_TAG),
        ('parameter', UINT32),
    )


class DWRITE_TYPOGRAPHIC_FEATURES(ctypes.Structure):
    _fields_ = (
        ('features', POINTER(DWRITE_FONT_FEATURE)),
        ('featureCount', UINT32),
    )


class DWRITE_SHAPING_TEXT_PROPERTIES(ctypes.Structure):
    _fields_ = (
        ('isShapedAlone', UINT16, 1),
        ('reserved1', UINT16, 1),
        ('canBreakShapingAfter', UINT16, 1),
        ('reserved', UINT16, 13),
    )

    def __repr__(self):
        return f"DWRITE_SHAPING_TEXT_PROPERTIES({self.isShapedAlone}, {self.reserved1}, {self.canBreakShapingAfter})"


class DWRITE_SHAPING_GLYPH_PROPERTIES(ctypes.Structure):
    _fields_ = (
        ('justification', UINT16, 4),
        ('isClusterStart', UINT16, 1),
        ('isDiacritic', UINT16, 1),
        ('isZeroWidthSpace', UINT16, 1),
        ('reserved', UINT16, 9),
    )


DWRITE_READING_DIRECTION = UINT
DWRITE_READING_DIRECTION_LEFT_TO_RIGHT = 0


class IDWriteTextAnalysisSource(com.IUnknown):
    _methods_ = [
        ('GetTextAtPosition',
         com.METHOD(HRESULT, c_void_p, UINT32, POINTER(c_wchar_p), POINTER(UINT32))),
        ('GetTextBeforePosition',
         com.STDMETHOD(UINT32, c_wchar_p, POINTER(UINT32))),
        ('GetParagraphReadingDirection',
         com.METHOD(DWRITE_READING_DIRECTION)),
        ('GetLocaleName',
         com.STDMETHOD(c_void_p, UINT32, POINTER(UINT32), POINTER(c_wchar_p))),
        ('GetNumberSubstitution',
         com.STDMETHOD(UINT32, POINTER(UINT32), c_void_p)),
    ]


class IDWriteTextAnalysisSink(com.IUnknown):
    _methods_ = [
        ('SetScriptAnalysis',
         com.STDMETHOD(c_void_p, UINT32, UINT32, POINTER(DWRITE_SCRIPT_ANALYSIS))),
        ('SetLineBreakpoints',
         com.STDMETHOD(UINT32, UINT32, c_void_p)),
        ('SetBidiLevel',
         com.STDMETHOD(UINT32, UINT32, UINT8, UINT8)),
        ('SetNumberSubstitution',
         com.STDMETHOD(UINT32, UINT32, c_void_p)),
    ]


class Run:
    def __init__(self):
        self.text_start = 0
        self.text_length = 0
        self.glyph_start = 0
        self.glyph_count = 0
        self.script = DWRITE_SCRIPT_ANALYSIS()
        self.bidi = 0
        self.isNumberSubstituted = False
        self.isSideways = False

        self.next_run = None

    def ContainsTextPosition(self, textPosition):
        return textPosition >= self.text_start and textPosition < self.text_start + self.text_length


class TextAnalysis(com.COMObject):
    _interfaces_ = [IDWriteTextAnalysisSource, IDWriteTextAnalysisSink]

    def __init__(self):
        super().__init__()
        self._textstart = 0
        self._textlength = 0
        self._glyphstart = 0
        self._glyphcount = 0
        self._ptrs = []

        self._script = None
        self._bidi = 0
        # self._sideways = False

    def GenerateResults(self, analyzer, text, text_length):
        self._text = text
        self._textstart = 0
        self._textlength = text_length
        self._glyphstart = 0
        self._glyphcount = 0

        self._start_run = Run()
        self._start_run.text_length = text_length

        self._current_run = self._start_run

        analyzer.AnalyzeScript(self, 0, text_length, self)

    def SetScriptAnalysis(self, this, textPosition, textLength, scriptAnalysis):
        # textPosition - The index of the first character in the string that the result applies to
        # textLength - How many characters of the string from the index that the result applies to
        # scriptAnalysis - The analysis information for all glyphs starting at position for length.
        self.SetCurrentRun(textPosition)
        self.SplitCurrentRun(textPosition)

        while textLength > 0:
            run, textLength = self.FetchNextRun(textLength)

            run.script.script = scriptAnalysis[0].script
            run.script.shapes = scriptAnalysis[0].shapes

            self._script = run.script

        return 0
        # return 0x80004001

    def GetTextBeforePosition(self, this):
        raise Exception("Currently not implemented.")

    def GetTextAtPosition(self, this, textPosition, textString, textLength):
        # This method will retrieve a substring of the text in this layout
        #   to be used in an analysis step.
        # Arguments:
        # textPosition - The index of the first character of the text to retrieve.
        # textString - The pointer to the first character of text at the index requested.
        # textLength - The characters available at/after the textString pointer (string length).
        if textPosition >= self._textlength:
            self._no_ptr = c_wchar_p(None)
            textString[0] = self._no_ptr
            textLength[0] = 0
        else:
            ptr = c_wchar_p(self._text[textPosition:])
            self._ptrs.append(ptr)
            textString[0] = ptr
            textLength[0] = self._textlength - textPosition

        return 0

    def GetParagraphReadingDirection(self):
        return 0

    def GetLocaleName(self, this, textPosition, textLength, localeName):
        self.__local_name = c_wchar_p("")  # TODO: Add more locales.
        localeName[0] = self.__local_name
        textLength[0] = self._textlength - textPosition
        return 0

    def GetNumberSubstitution(self):
        return 0

    def SetCurrentRun(self, textPosition):
        if self._current_run and self._current_run.ContainsTextPosition(textPosition):
            return

    def SplitCurrentRun(self, textPosition):
        if not self._current_run:
            return

        if textPosition <= self._current_run.text_start:
            # Already first start of the run.
            return

        new_run = copy.copy(self._current_run)

        new_run.next_run = self._current_run.next_run
        self._current_run.next_run = new_run

        splitPoint = textPosition - self._current_run.text_start
        new_run.text_start += splitPoint
        new_run.text_length -= splitPoint

        self._current_run.text_length = splitPoint
        self._current_run = new_run

    def FetchNextRun(self, textLength):
        original_run = self._current_run

        if (textLength < self._current_run.text_length):
            self.SplitCurrentRun(self._current_run.text_start + textLength)
        else:
            self._current_run = self._current_run.next_run

        textLength -= original_run.text_length

        return original_run, textLength


class IDWriteTextAnalyzer(com.pIUnknown):
    _methods_ = [
        ('AnalyzeScript',
         com.STDMETHOD(POINTER(IDWriteTextAnalysisSource), UINT32, UINT32, POINTER(IDWriteTextAnalysisSink))),
        ('AnalyzeBidi',
         com.STDMETHOD()),
        ('AnalyzeNumberSubstitution',
         com.STDMETHOD()),
        ('AnalyzeLineBreakpoints',
         com.STDMETHOD()),
        ('GetGlyphs',
         com.STDMETHOD(c_wchar_p, UINT32, IDWriteFontFace, BOOL, BOOL, POINTER(DWRITE_SCRIPT_ANALYSIS),
                       c_wchar_p, c_void_p, POINTER(POINTER(DWRITE_TYPOGRAPHIC_FEATURES)), POINTER(UINT32),
                       UINT32, UINT32, POINTER(UINT16), POINTER(DWRITE_SHAPING_TEXT_PROPERTIES),
                       POINTER(UINT16), POINTER(DWRITE_SHAPING_GLYPH_PROPERTIES), POINTER(UINT32))),
        ('GetGlyphPlacements',
         com.STDMETHOD(c_wchar_p, POINTER(UINT16), POINTER(DWRITE_SHAPING_TEXT_PROPERTIES), UINT32, POINTER(UINT16),
                       POINTER(DWRITE_SHAPING_GLYPH_PROPERTIES), UINT32, IDWriteFontFace, FLOAT, BOOL, BOOL,
                       POINTER(DWRITE_SCRIPT_ANALYSIS), c_wchar_p, POINTER(DWRITE_TYPOGRAPHIC_FEATURES),
                       POINTER(UINT32), UINT32, POINTER(FLOAT), POINTER(DWRITE_GLYPH_OFFSET))),
        ('GetGdiCompatibleGlyphPlacements',
         com.STDMETHOD()),
    ]


class IDWriteLocalizedStrings(com.pIUnknown):
    _methods_ = [
        ('GetCount',
         com.METHOD(UINT32)),
        ('FindLocaleName',
         com.STDMETHOD(c_wchar_p, POINTER(UINT32), POINTER(BOOL))),
        ('GetLocaleNameLength',
         com.STDMETHOD(UINT32, POINTER(UINT32))),
        ('GetLocaleName',
         com.STDMETHOD(UINT32, c_wchar_p, UINT32)),
        ('GetStringLength',
         com.STDMETHOD(UINT32, POINTER(UINT32))),
        ('GetString',
         com.STDMETHOD(UINT32, c_wchar_p, UINT32)),
    ]


class IDWriteFontList(com.pIUnknown):
    _methods_ = [
        ('GetFontCollection',
         com.STDMETHOD()),
        ('GetFontCount',
         com.STDMETHOD()),
        ('GetFont',
         com.STDMETHOD()),
    ]


class IDWriteFontFamily(IDWriteFontList, com.pIUnknown):
    _methods_ = [
        ('GetFamilyNames',
         com.STDMETHOD(POINTER(IDWriteLocalizedStrings))),
        ('GetFirstMatchingFont',
         com.STDMETHOD(DWRITE_FONT_WEIGHT, DWRITE_FONT_STRETCH, DWRITE_FONT_STYLE, c_void_p)),
        ('GetMatchingFonts',
         com.STDMETHOD()),
    ]


class IDWriteFontFamily1(IDWriteFontFamily, IDWriteFontList, com.pIUnknown):
    _methods_ = [
        ('GetFontLocality',
         com.STDMETHOD()),
        ('GetFont1',
         com.STDMETHOD()),
        ('GetFontFaceReference',
         com.STDMETHOD()),
    ]


class IDWriteFontFile(com.pIUnknown):
    _methods_ = [
        ('GetReferenceKey',
         com.STDMETHOD()),
        ('GetLoader',
         com.STDMETHOD()),
        ('Analyze',
         com.STDMETHOD()),
    ]


class IDWriteFont(com.pIUnknown):
    _methods_ = [
        ('GetFontFamily',
         com.STDMETHOD(POINTER(IDWriteFontFamily))),
        ('GetWeight',
         com.STDMETHOD()),
        ('GetStretch',
         com.STDMETHOD()),
        ('GetStyle',
         com.STDMETHOD()),
        ('IsSymbolFont',
         com.STDMETHOD()),
        ('GetFaceNames',
         com.STDMETHOD(POINTER(IDWriteLocalizedStrings))),
        ('GetInformationalStrings',
         com.STDMETHOD()),
        ('GetSimulations',
         com.STDMETHOD()),
        ('GetMetrics',
         com.STDMETHOD()),
        ('HasCharacter',
         com.STDMETHOD(UINT32, POINTER(BOOL))),
        ('CreateFontFace',
         com.STDMETHOD(POINTER(IDWriteFontFace))),
    ]


class IDWriteFont1(IDWriteFont, com.pIUnknown):
    _methods_ = [
        ('GetMetrics1',
         com.STDMETHOD()),
        ('GetPanose',
         com.STDMETHOD()),
        ('GetUnicodeRanges',
         com.STDMETHOD()),
        ('IsMonospacedFont',
         com.STDMETHOD())
    ]


class IDWriteFontCollection(com.pIUnknown):
    _methods_ = [
        ('GetFontFamilyCount',
         com.STDMETHOD()),
        ('GetFontFamily',
         com.STDMETHOD(UINT32, POINTER(IDWriteFontFamily))),
        ('FindFamilyName',
         com.STDMETHOD(c_wchar_p, POINTER(UINT), POINTER(BOOL))),
        ('GetFontFromFontFace',
         com.STDMETHOD()),
    ]


class IDWriteFontCollection1(IDWriteFontCollection, com.pIUnknown):
    _methods_ = [
        ('GetFontSet',
         com.STDMETHOD()),
        ('GetFontFamily1',
         com.STDMETHOD(POINTER(IDWriteFontFamily1))),
    ]


DWRITE_TEXT_ALIGNMENT = UINT
DWRITE_TEXT_ALIGNMENT_LEADING = 1
DWRITE_TEXT_ALIGNMENT_TRAILING = 2
DWRITE_TEXT_ALIGNMENT_CENTER = 3
DWRITE_TEXT_ALIGNMENT_JUSTIFIED = 4


class IDWriteTextFormat(com.pIUnknown):
    _methods_ = [
        ('SetTextAlignment',
         com.STDMETHOD(DWRITE_TEXT_ALIGNMENT)),
        ('SetParagraphAlignment',
         com.STDMETHOD()),
        ('SetWordWrapping',
         com.STDMETHOD()),
        ('SetReadingDirection',
         com.STDMETHOD()),
        ('SetFlowDirection',
         com.STDMETHOD()),
        ('SetIncrementalTabStop',
         com.STDMETHOD()),
        ('SetTrimming',
         com.STDMETHOD()),
        ('SetLineSpacing',
         com.STDMETHOD()),
        ('GetTextAlignment',
         com.STDMETHOD()),
        ('GetParagraphAlignment',
         com.STDMETHOD()),
        ('GetWordWrapping',
         com.STDMETHOD()),
        ('GetReadingDirection',
         com.STDMETHOD()),
        ('GetFlowDirection',
         com.STDMETHOD()),
        ('GetIncrementalTabStop',
         com.STDMETHOD()),
        ('GetTrimming',
         com.STDMETHOD()),
        ('GetLineSpacing',
         com.STDMETHOD()),
        ('GetFontCollection',
         com.STDMETHOD()),
        ('GetFontFamilyNameLength',
         com.STDMETHOD()),
        ('GetFontFamilyName',
         com.STDMETHOD()),
        ('GetFontWeight',
         com.STDMETHOD()),
        ('GetFontStyle',
         com.STDMETHOD()),
        ('GetFontStretch',
         com.STDMETHOD()),
        ('GetFontSize',
         com.STDMETHOD()),
        ('GetLocaleNameLength',
         com.STDMETHOD()),
        ('GetLocaleName',
         com.STDMETHOD()),
    ]


class IDWriteTypography(com.pIUnknown):
    _methods_ = [
        ('AddFontFeature',
         com.STDMETHOD(DWRITE_FONT_FEATURE)),
        ('GetFontFeatureCount',
         com.METHOD(UINT32)),
        ('GetFontFeature',
         com.STDMETHOD())
    ]


class DWRITE_TEXT_RANGE(ctypes.Structure):
    _fields_ = (
        ('startPosition', UINT32),
        ('length', UINT32),
    )


class DWRITE_OVERHANG_METRICS(ctypes.Structure):
    _fields_ = (
        ('left', FLOAT),
        ('top', FLOAT),
        ('right', FLOAT),
        ('bottom', FLOAT),
    )


class IDWriteTextLayout(IDWriteTextFormat, com.pIUnknown):
    _methods_ = [
        ('SetMaxWidth',
         com.STDMETHOD()),
        ('SetMaxHeight',
         com.STDMETHOD()),
        ('SetFontCollection',
         com.STDMETHOD()),
        ('SetFontFamilyName',
         com.STDMETHOD()),
        ('SetFontWeight',  # 30
         com.STDMETHOD()),
        ('SetFontStyle',
         com.STDMETHOD()),
        ('SetFontStretch',
         com.STDMETHOD()),
        ('SetFontSize',
         com.STDMETHOD()),
        ('SetUnderline',
         com.STDMETHOD()),
        ('SetStrikethrough',
         com.STDMETHOD()),
        ('SetDrawingEffect',
         com.STDMETHOD()),
        ('SetInlineObject',
         com.STDMETHOD()),
        ('SetTypography',
         com.STDMETHOD(IDWriteTypography, DWRITE_TEXT_RANGE)),
        ('SetLocaleName',
         com.STDMETHOD()),
        ('GetMaxWidth',  # 40
         com.METHOD(FLOAT)),
        ('GetMaxHeight',
         com.METHOD(FLOAT)),
        ('GetFontCollection2',
         com.STDMETHOD()),
        ('GetFontFamilyNameLength2',
         com.STDMETHOD()),
        ('GetFontFamilyName2',
         com.STDMETHOD()),
        ('GetFontWeight2',
         com.STDMETHOD(UINT32, POINTER(DWRITE_FONT_WEIGHT), POINTER(DWRITE_TEXT_RANGE))),
        ('GetFontStyle2',
         com.STDMETHOD()),
        ('GetFontStretch2',
         com.STDMETHOD()),
        ('GetFontSize2',
         com.STDMETHOD()),
        ('GetUnderline',
         com.STDMETHOD()),
        ('GetStrikethrough',
         com.STDMETHOD(UINT32, POINTER(BOOL), POINTER(DWRITE_TEXT_RANGE))),
        ('GetDrawingEffect',
         com.STDMETHOD()),
        ('GetInlineObject',
         com.STDMETHOD()),
        ('GetTypography',  # Always returns NULL without SetTypography being called.
         com.STDMETHOD(UINT32, POINTER(IDWriteTypography), POINTER(DWRITE_TEXT_RANGE))),
        ('GetLocaleNameLength1',
         com.STDMETHOD()),
        ('GetLocaleName1',
         com.STDMETHOD()),
        ('Draw',
         com.STDMETHOD()),
        ('GetLineMetrics',
         com.STDMETHOD()),
        ('GetMetrics',
         com.STDMETHOD(POINTER(DWRITE_TEXT_METRICS))),
        ('GetOverhangMetrics',
         com.STDMETHOD(POINTER(DWRITE_OVERHANG_METRICS))),
        ('GetClusterMetrics',
         com.STDMETHOD(POINTER(DWRITE_CLUSTER_METRICS), UINT32, POINTER(UINT32))),
        ('DetermineMinWidth',
         com.STDMETHOD(POINTER(FLOAT))),
        ('HitTestPoint',
         com.STDMETHOD()),
        ('HitTestTextPosition',
         com.STDMETHOD()),
        ('HitTestTextRange',
         com.STDMETHOD()),
    ]


class IDWriteTextLayout1(IDWriteTextLayout, IDWriteTextFormat, com.pIUnknown):
    _methods_ = [
        ('SetPairKerning',
         com.STDMETHOD()),
        ('GetPairKerning',
         com.STDMETHOD()),
        ('SetCharacterSpacing',
         com.STDMETHOD()),
        ('GetCharacterSpacing',
         com.STDMETHOD(UINT32, POINTER(FLOAT), POINTER(FLOAT), POINTER(FLOAT), POINTER(DWRITE_TEXT_RANGE))),
    ]


class IDWriteFontFileEnumerator(com.IUnknown):
    _methods_ = [
        ('MoveNext',
         com.STDMETHOD(c_void_p, POINTER(BOOL))),
        ('GetCurrentFontFile',
         com.STDMETHOD(c_void_p, c_void_p)),
    ]


class IDWriteFontCollectionLoader(com.IUnknown):
    _methods_ = [
        ('CreateEnumeratorFromKey',
         com.STDMETHOD(c_void_p, c_void_p, c_void_p, UINT32, POINTER(POINTER(IDWriteFontFileEnumerator)))),
    ]


class IDWriteFontFileStream(com.IUnknown):
    _methods_ = [
        ('ReadFileFragment',
         com.STDMETHOD(c_void_p, POINTER(c_void_p), UINT64, UINT64, POINTER(c_void_p))),
        ('ReleaseFileFragment',
         com.STDMETHOD(c_void_p, c_void_p)),
        ('GetFileSize',
         com.STDMETHOD(c_void_p, POINTER(UINT64))),
        ('GetLastWriteTime',
         com.STDMETHOD(c_void_p, POINTER(UINT64))),
    ]


class MyFontFileStream(com.COMObject):
    _interfaces_ = [IDWriteFontFileStream]

    def __init__(self, data):
        self._data = data
        self._size = len(data)
        self._ptrs = []

    def AddRef(self, this):
        return 1

    def Release(self, this):
        return 1

    def QueryInterface(self, this, refiid, tester):
        return 0

    def ReadFileFragment(self, this, fragmentStart, fileOffset, fragmentSize, fragmentContext):
        if fileOffset + fragmentSize > self._size:
            return 0x80004005  # E_FAIL

        fragment = self._data[fileOffset:]
        buffer = (ctypes.c_ubyte * len(fragment)).from_buffer(bytearray(fragment))
        ptr = cast(buffer, c_void_p)

        self._ptrs.append(ptr)
        fragmentStart[0] = ptr
        fragmentContext[0] = None
        return 0

    def ReleaseFileFragment(self, this, fragmentContext):
        return 0

    def GetFileSize(self, this, fileSize):
        fileSize[0] = self._size
        return 0

    def GetLastWriteTime(self, this, lastWriteTime):
        return 0x80004001  # E_NOTIMPL


class IDWriteFontFileLoader(com.IUnknown):
    _methods_ = [
        ('CreateStreamFromKey',
         com.STDMETHOD(c_void_p, c_void_p, UINT32, POINTER(POINTER(IDWriteFontFileStream))))
    ]


class LegacyFontFileLoader(com.COMObject):
    _interfaces_ = [IDWriteFontFileLoader]

    def __init__(self):
        self._streams = {}

    def QueryInterface(self, this, refiid, tester):
        return 0

    def AddRef(self, this):
        return 1

    def Release(self, this):
        return 1

    def CreateStreamFromKey(self, this, fontfileReferenceKey, fontFileReferenceKeySize, fontFileStream):
        convert_index = cast(fontfileReferenceKey, POINTER(c_uint32))

        self._ptr = ctypes.cast(self._streams[convert_index.contents.value]._pointers[IDWriteFontFileStream],
                                POINTER(IDWriteFontFileStream))
        fontFileStream[0] = self._ptr
        return 0

    def SetCurrentFont(self, index, data):
        self._streams[index] = MyFontFileStream(data)


class MyEnumerator(com.COMObject):
    _interfaces_ = [IDWriteFontFileEnumerator]

    def __init__(self, factory, loader):
        self.factory = cast(factory, IDWriteFactory)
        self.key = "pyglet_dwrite"
        self.size = len(self.key)
        self.current_index = -1

        self._keys = []
        self._font_data = []
        self._font_files = []
        self._current_file = None

        self._font_key_ref = create_unicode_buffer("none")
        self._font_key_len = len(self._font_key_ref)

        self._file_loader = loader

    def AddFontData(self, fonts):
        self._font_data = fonts

    def MoveNext(self, this, hasCurrentFile):

        self.current_index += 1
        if self.current_index != len(self._font_data):
            font_file = IDWriteFontFile()

            self._file_loader.SetCurrentFont(self.current_index, self._font_data[self.current_index])

            key = self.current_index

            if not self.current_index in self._keys:
                buffer = pointer(c_uint32(key))

                ptr = cast(buffer, c_void_p)

                self._keys.append(ptr)

            self.factory.CreateCustomFontFileReference(self._keys[self.current_index],
                                                       sizeof(buffer),
                                                       self._file_loader,
                                                       byref(font_file))

            self._font_files.append(font_file)

            hasCurrentFile[0] = 1
        else:
            hasCurrentFile[0] = 0

        pass

    def GetCurrentFontFile(self, this, fontFile):
        fontFile = cast(fontFile, POINTER(IDWriteFontFile))
        fontFile[0] = self._font_files[self.current_index]
        return 0


class LegacyCollectionLoader(com.COMObject):
    _interfaces_ = [IDWriteFontCollectionLoader]

    def __init__(self, factory, loader):
        self._enumerator = MyEnumerator(factory, loader)

    def AddFontData(self, fonts):
        self._enumerator.AddFontData(fonts)

    def AddRef(self, this):
        self._i = 1
        return 1

    def Release(self, this):
        self._i = 0
        return 1

    def QueryInterface(self, this, refiid, tester):
        return 0

    def CreateEnumeratorFromKey(self, this, factory, key, key_size, enumerator):
        self._ptr = ctypes.cast(self._enumerator._pointers[IDWriteFontFileEnumerator],
                                POINTER(IDWriteFontFileEnumerator))

        enumerator[0] = self._ptr
        return 0


IID_IDWriteFactory = com.GUID(0xb859ee5a, 0xd838, 0x4b5b, 0xa2, 0xe8, 0x1a, 0xdc, 0x7d, 0x93, 0xdb, 0x48)


class IDWriteFactory(com.pIUnknown):
    _methods_ = [
        ('GetSystemFontCollection',
         com.STDMETHOD(POINTER(IDWriteFontCollection), BOOL)),
        ('CreateCustomFontCollection',
         com.STDMETHOD(POINTER(IDWriteFontCollectionLoader), c_void_p, UINT32, POINTER(IDWriteFontCollection))),
        ('RegisterFontCollectionLoader',
         com.STDMETHOD(POINTER(IDWriteFontCollectionLoader))),
        ('UnregisterFontCollectionLoader',
         com.STDMETHOD(POINTER(IDWriteFontCollectionLoader))),
        ('CreateFontFileReference',
         com.STDMETHOD(c_wchar_p, c_void_p, POINTER(IDWriteFontFile))),
        ('CreateCustomFontFileReference',
         com.STDMETHOD(c_void_p, UINT32, POINTER(IDWriteFontFileLoader), POINTER(IDWriteFontFile))),
        ('CreateFontFace',
         com.STDMETHOD()),
        ('CreateRenderingParams',
         com.STDMETHOD()),
        ('CreateMonitorRenderingParams',
         com.STDMETHOD()),
        ('CreateCustomRenderingParams',
         com.STDMETHOD()),
        ('RegisterFontFileLoader',
         com.STDMETHOD(c_void_p)),  # Ambigious as newer is a pIUnknown and legacy is IUnknown.
        ('UnregisterFontFileLoader',
         com.STDMETHOD(POINTER(IDWriteFontFileLoader))),
        ('CreateTextFormat',
         com.STDMETHOD(c_wchar_p, IDWriteFontCollection, DWRITE_FONT_WEIGHT, DWRITE_FONT_STYLE, DWRITE_FONT_STRETCH,
                       FLOAT, c_wchar_p, POINTER(IDWriteTextFormat))),
        ('CreateTypography',
         com.STDMETHOD(POINTER(IDWriteTypography))),
        ('GetGdiInterop',
         com.STDMETHOD()),
        ('CreateTextLayout',
         com.STDMETHOD(c_wchar_p, UINT32, IDWriteTextFormat, FLOAT, FLOAT, POINTER(IDWriteTextLayout))),
        ('CreateGdiCompatibleTextLayout',
         com.STDMETHOD()),
        ('CreateEllipsisTrimmingSign',
         com.STDMETHOD()),
        ('CreateTextAnalyzer',
         com.STDMETHOD(POINTER(IDWriteTextAnalyzer))),
        ('CreateNumberSubstitution',
         com.STDMETHOD()),
        ('CreateGlyphRunAnalysis',
         com.STDMETHOD()),
    ]


IID_IDWriteFactory1 = com.GUID(0x30572f99, 0xdac6, 0x41db, 0xa1, 0x6e, 0x04, 0x86, 0x30, 0x7e, 0x60, 0x6a)


class IDWriteFactory1(IDWriteFactory, com.pIUnknown):
    _methods_ = [
        ('GetEudcFontCollection',
         com.STDMETHOD()),
        ('CreateCustomRenderingParams',
         com.STDMETHOD()),
    ]


class IDWriteFontFallback(com.pIUnknown):
    _methods_ = [
        ('MapCharacters',
         com.STDMETHOD(POINTER(IDWriteTextAnalysisSource), UINT32, UINT32, POINTER(IDWriteFontCollection), c_wchar_p,
                       DWRITE_FONT_WEIGHT, DWRITE_FONT_STYLE, DWRITE_FONT_STRETCH, POINTER(UINT32),
                       POINTER(IDWriteFont),
                       POINTER(FLOAT))),
    ]


class IDWriteFactory2(IDWriteFactory1, com.pIUnknown):
    _methods_ = [
        ('GetSystemFontFallback',
         com.STDMETHOD(POINTER(IDWriteFontFallback))),
        ('CreateFontFallbackBuilder',
         com.STDMETHOD()),
        ('TranslateColorGlyphRun',
         com.STDMETHOD()),
        ('CreateCustomRenderingParams',
         com.STDMETHOD()),
        ('CreateGlyphRunAnalysis',
         com.STDMETHOD()),
    ]


class IDWriteFontSet(com.pIUnknown):
    _methods_ = [
        ('GetFontCount',
         com.STDMETHOD()),
        ('GetFontFaceReference',
         com.STDMETHOD()),
        ('FindFontFaceReference',
         com.STDMETHOD()),
        ('FindFontFace',
         com.STDMETHOD()),
        ('GetPropertyValues',
         com.STDMETHOD()),
        ('GetPropertyOccurrenceCount',
         com.STDMETHOD()),
        ('GetMatchingFonts',
         com.STDMETHOD()),
        ('GetMatchingFonts',
         com.STDMETHOD()),
    ]


class IDWriteFontSetBuilder(com.pIUnknown):
    _methods_ = [
        ('AddFontFaceReference',
         com.STDMETHOD()),
        ('AddFontFaceReference',
         com.STDMETHOD()),
        ('AddFontSet',
         com.STDMETHOD()),
        ('CreateFontSet',
         com.STDMETHOD(POINTER(IDWriteFontSet))),
    ]


class IDWriteFontSetBuilder1(IDWriteFontSetBuilder, com.pIUnknown):
    _methods_ = [
        ('AddFontFile',
         com.STDMETHOD(IDWriteFontFile)),
    ]


class IDWriteFactory3(IDWriteFactory2, com.pIUnknown):
    _methods_ = [
        ('CreateGlyphRunAnalysis',
         com.STDMETHOD()),
        ('CreateCustomRenderingParams',
         com.STDMETHOD()),
        ('CreateFontFaceReference',
         com.STDMETHOD()),
        ('CreateFontFaceReference',
         com.STDMETHOD()),
        ('GetSystemFontSet',
         com.STDMETHOD()),
        ('CreateFontSetBuilder',
         com.STDMETHOD(POINTER(IDWriteFontSetBuilder))),
        ('CreateFontCollectionFromFontSet',
         com.STDMETHOD(IDWriteFontSet, POINTER(IDWriteFontCollection1))),
        ('GetSystemFontCollection3',
         com.STDMETHOD()),
        ('GetFontDownloadQueue',
         com.STDMETHOD()),
        ('GetSystemFontSet',
         com.STDMETHOD()),
    ]


class IDWriteFactory4(IDWriteFactory3, com.pIUnknown):
    _methods_ = [
        ('TranslateColorGlyphRun',
         com.STDMETHOD()),
        ('ComputeGlyphOrigins',
         com.STDMETHOD()),
    ]


class IDWriteInMemoryFontFileLoader(com.pIUnknown):
    _methods_ = [
        ('CreateStreamFromKey',
         com.STDMETHOD()),
        ('CreateInMemoryFontFileReference',
         com.STDMETHOD(IDWriteFactory, c_void_p, UINT, c_void_p, POINTER(IDWriteFontFile))),
        ('GetFileCount',
         com.STDMETHOD()),
    ]


IID_IDWriteFactory5 = com.GUID(0x958DB99A, 0xBE2A, 0x4F09, 0xAF, 0x7D, 0x65, 0x18, 0x98, 0x03, 0xD1, 0xD3)


class IDWriteFactory5(IDWriteFactory4, IDWriteFactory3, IDWriteFactory2, IDWriteFactory1, IDWriteFactory,
                      com.pIUnknown):
    _methods_ = [
        ('CreateFontSetBuilder1',
         com.STDMETHOD(POINTER(IDWriteFontSetBuilder1))),
        ('CreateInMemoryFontFileLoader',
         com.STDMETHOD(POINTER(IDWriteInMemoryFontFileLoader))),
        ('CreateHttpFontFileLoader',
         com.STDMETHOD()),
        ('AnalyzeContainerType',
         com.STDMETHOD())
    ]


class DWRITE_GLYPH_RUN(ctypes.Structure):
    _fields_ = (
        ('fontFace', IDWriteFontFace),
        ('fontEmSize', FLOAT),
        ('glyphCount', UINT32),
        ('glyphIndices', POINTER(UINT16)),
        ('glyphAdvances', POINTER(FLOAT)),
        ('glyphOffsets', POINTER(DWRITE_GLYPH_OFFSET)),
        ('isSideways', BOOL),
        ('bidiLevel', UINT32),
    )


DWriteCreateFactory = dwrite_lib.DWriteCreateFactory
DWriteCreateFactory.restype = HRESULT
DWriteCreateFactory.argtypes = [DWRITE_FACTORY_TYPE, com.REFIID, POINTER(com.pIUnknown)]


class D2D_POINT_2F(Structure):
    _fields_ = (
        ('x', FLOAT),
        ('y', FLOAT),
    )


class D2D1_RECT_F(Structure):
    _fields_ = (
        ('left', FLOAT),
        ('top', FLOAT),
        ('right', FLOAT),
        ('bottom', FLOAT),
    )


class D2D1_COLOR_F(Structure):
    _fields_ = (
        ('r', FLOAT),
        ('g', FLOAT),
        ('b', FLOAT),
        ('a', FLOAT),
    )


class ID2D1Resource(com.pIUnknown):
    _methods_ = [
        ('GetFactory',
         com.STDMETHOD()),
    ]


class ID2D1Brush(ID2D1Resource, com.pIUnknown):
    _methods_ = [
        ('SetOpacity',
         com.STDMETHOD()),
        ('SetTransform',
         com.STDMETHOD()),
        ('GetOpacity',
         com.STDMETHOD()),
        ('GetTransform',
         com.STDMETHOD()),
    ]


class ID2D1SolidColorBrush(ID2D1Brush, ID2D1Resource, com.pIUnknown):
    _methods_ = [
        ('SetColor',
         com.STDMETHOD()),
        ('GetColor',
         com.STDMETHOD()),
    ]


D2D1_TEXT_ANTIALIAS_MODE = UINT
D2D1_TEXT_ANTIALIAS_MODE_DEFAULT = 0
D2D1_TEXT_ANTIALIAS_MODE_CLEARTYPE = 1
D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE = 2
D2D1_TEXT_ANTIALIAS_MODE_ALIASED = 3

D2D1_RENDER_TARGET_TYPE = UINT
D2D1_RENDER_TARGET_TYPE_DEFAULT = 0
D2D1_RENDER_TARGET_TYPE_SOFTWARE = 1
D2D1_RENDER_TARGET_TYPE_HARDWARE = 2

D2D1_FEATURE_LEVEL = UINT
D2D1_FEATURE_LEVEL_DEFAULT = 0

D2D1_RENDER_TARGET_USAGE = UINT
D2D1_RENDER_TARGET_USAGE_NONE = 0
D2D1_RENDER_TARGET_USAGE_FORCE_BITMAP_REMOTING = 1
D2D1_RENDER_TARGET_USAGE_GDI_COMPATIBLE = 2

DXGI_FORMAT = UINT
DXGI_FORMAT_UNKNOWN = 0

D2D1_ALPHA_MODE = UINT
D2D1_ALPHA_MODE_UNKNOWN = 0
D2D1_ALPHA_MODE_PREMULTIPLIED = 1
D2D1_ALPHA_MODE_STRAIGHT = 2
D2D1_ALPHA_MODE_IGNORE = 3

D2D1_DRAW_TEXT_OPTIONS = UINT
D2D1_DRAW_TEXT_OPTIONS_NO_SNAP = 0x00000001
D2D1_DRAW_TEXT_OPTIONS_CLIP = 0x00000002
D2D1_DRAW_TEXT_OPTIONS_ENABLE_COLOR_FONT = 0x00000004
D2D1_DRAW_TEXT_OPTIONS_DISABLE_COLOR_BITMAP_SNAPPING = 0x00000008
D2D1_DRAW_TEXT_OPTIONS_NONE = 0x00000000
D2D1_DRAW_TEXT_OPTIONS_FORCE_DWORD = 0xffffffff

DWRITE_MEASURING_MODE = UINT
DWRITE_MEASURING_MODE_NATURAL = 0
DWRITE_MEASURING_MODE_GDI_CLASSIC = 1
DWRITE_MEASURING_MODE_GDI_NATURAL = 2


class D2D1_PIXEL_FORMAT(Structure):
    _fields_ = (
        ('format', DXGI_FORMAT),
        ('alphaMode', D2D1_ALPHA_MODE),
    )


class D2D1_RENDER_TARGET_PROPERTIES(Structure):
    _fields_ = (
        ('type', D2D1_RENDER_TARGET_TYPE),
        ('pixelFormat', D2D1_PIXEL_FORMAT),
        ('dpiX', FLOAT),
        ('dpiY', FLOAT),
        ('usage', D2D1_RENDER_TARGET_USAGE),
        ('minLevel', D2D1_FEATURE_LEVEL),
    )


DXGI_FORMAT_B8G8R8A8_UNORM = 87

pixel_format = D2D1_PIXEL_FORMAT()
pixel_format.format = DXGI_FORMAT_UNKNOWN
pixel_format.alphaMode = D2D1_ALPHA_MODE_UNKNOWN

default_target_properties = D2D1_RENDER_TARGET_PROPERTIES()
default_target_properties.type = D2D1_RENDER_TARGET_TYPE_DEFAULT
default_target_properties.pixelFormat = pixel_format
default_target_properties.dpiX = 0.0
default_target_properties.dpiY = 0.0
default_target_properties.usage = D2D1_RENDER_TARGET_USAGE_NONE
default_target_properties.minLevel = D2D1_FEATURE_LEVEL_DEFAULT


class ID2D1RenderTarget(ID2D1Resource, com.pIUnknown):
    _methods_ = [
        ('CreateBitmap',
         com.STDMETHOD()),
        ('CreateBitmapFromWicBitmap',
         com.STDMETHOD()),
        ('CreateSharedBitmap',
         com.STDMETHOD()),
        ('CreateBitmapBrush',
         com.STDMETHOD()),
        ('CreateSolidColorBrush',
         com.STDMETHOD(POINTER(D2D1_COLOR_F), c_void_p, POINTER(ID2D1SolidColorBrush))),
        ('CreateGradientStopCollection',
         com.STDMETHOD()),
        ('CreateLinearGradientBrush',
         com.STDMETHOD()),
        ('CreateRadialGradientBrush',
         com.STDMETHOD()),
        ('CreateCompatibleRenderTarget',
         com.STDMETHOD()),
        ('CreateLayer',
         com.STDMETHOD()),
        ('CreateMesh',
         com.STDMETHOD()),
        ('DrawLine',
         com.STDMETHOD()),
        ('DrawRectangle',
         com.STDMETHOD()),
        ('FillRectangle',
         com.STDMETHOD()),
        ('DrawRoundedRectangle',
         com.STDMETHOD()),
        ('FillRoundedRectangle',
         com.STDMETHOD()),
        ('DrawEllipse',
         com.STDMETHOD()),
        ('FillEllipse',
         com.STDMETHOD()),
        ('DrawGeometry',
         com.STDMETHOD()),
        ('FillGeometry',
         com.STDMETHOD()),
        ('FillMesh',
         com.STDMETHOD()),
        ('FillOpacityMask',
         com.STDMETHOD()),
        ('DrawBitmap',
         com.STDMETHOD()),
        ('DrawText',
         com.STDMETHOD(c_wchar_p, UINT, IDWriteTextFormat, POINTER(D2D1_RECT_F), ID2D1Brush, D2D1_DRAW_TEXT_OPTIONS,
                       DWRITE_MEASURING_MODE)),
        ('DrawTextLayout',
         com.METHOD(c_void, D2D_POINT_2F, IDWriteTextLayout, ID2D1Brush, UINT32)),
        ('DrawGlyphRun',
         com.METHOD(c_void, D2D_POINT_2F, POINTER(DWRITE_GLYPH_RUN), ID2D1Brush, UINT32)),
        ('SetTransform',
         com.METHOD(c_void)),
        ('GetTransform',
         com.STDMETHOD()),
        ('SetAntialiasMode',
         com.STDMETHOD()),
        ('GetAntialiasMode',
         com.STDMETHOD()),
        ('SetTextAntialiasMode',
         com.METHOD(c_void, D2D1_TEXT_ANTIALIAS_MODE)),
        ('GetTextAntialiasMode',
         com.STDMETHOD()),
        ('SetTextRenderingParams',
         com.STDMETHOD()),
        ('GetTextRenderingParams',
         com.STDMETHOD()),
        ('SetTags',
         com.STDMETHOD()),
        ('GetTags',
         com.STDMETHOD()),
        ('PushLayer',
         com.STDMETHOD()),
        ('PopLayer',
         com.STDMETHOD()),
        ('Flush',
         com.STDMETHOD()),
        ('SaveDrawingState',
         com.STDMETHOD()),
        ('RestoreDrawingState',
         com.STDMETHOD()),
        ('PushAxisAlignedClip',
         com.STDMETHOD()),
        ('PopAxisAlignedClip',
         com.STDMETHOD()),
        ('Clear',
         com.METHOD(c_void, POINTER(D2D1_COLOR_F))),
        ('BeginDraw',
         com.METHOD(c_void)),
        ('EndDraw',
         com.STDMETHOD(c_void_p, c_void_p)),
        ('GetPixelFormat',
         com.STDMETHOD()),
        ('SetDpi',
         com.STDMETHOD()),
        ('GetDpi',
         com.STDMETHOD()),
        ('GetSize',
         com.STDMETHOD()),
        ('GetPixelSize',
         com.STDMETHOD()),
        ('GetMaximumBitmapSize',
         com.STDMETHOD()),
        ('IsSupported',
         com.STDMETHOD()),
    ]


IID_ID2D1Factory = com.GUID(0x06152247, 0x6f50, 0x465a, 0x92, 0x45, 0x11, 0x8b, 0xfd, 0x3b, 0x60, 0x07)


class ID2D1Factory(com.pIUnknown):
    _methods_ = [
        ('ReloadSystemMetrics',
         com.STDMETHOD()),
        ('GetDesktopDpi',
         com.STDMETHOD()),
        ('CreateRectangleGeometry',
         com.STDMETHOD()),
        ('CreateRoundedRectangleGeometry',
         com.STDMETHOD()),
        ('CreateEllipseGeometry',
         com.STDMETHOD()),
        ('CreateGeometryGroup',
         com.STDMETHOD()),
        ('CreateTransformedGeometry',
         com.STDMETHOD()),
        ('CreatePathGeometry',
         com.STDMETHOD()),
        ('CreateStrokeStyle',
         com.STDMETHOD()),
        ('CreateDrawingStateBlock',
         com.STDMETHOD()),
        ('CreateWicBitmapRenderTarget',
         com.STDMETHOD(IWICBitmap, POINTER(D2D1_RENDER_TARGET_PROPERTIES), POINTER(ID2D1RenderTarget))),
        ('CreateHwndRenderTarget',
         com.STDMETHOD()),
        ('CreateDxgiSurfaceRenderTarget',
         com.STDMETHOD()),
        ('CreateDCRenderTarget',
         com.STDMETHOD()),
    ]


d2d_lib = ctypes.windll.d2d1

D2D1_FACTORY_TYPE = UINT
D2D1_FACTORY_TYPE_SINGLE_THREADED = 0
D2D1_FACTORY_TYPE_MULTI_THREADED = 1

D2D1CreateFactory = d2d_lib.D2D1CreateFactory
D2D1CreateFactory.restype = HRESULT
D2D1CreateFactory.argtypes = [D2D1_FACTORY_TYPE, com.REFIID, c_void_p, c_void_p]

# We need a WIC factory to make this work. Make sure one is in the initialized decoders.
wic_decoder = None
for decoder in pyglet.image.codecs.get_decoders():
    if isinstance(decoder, WICDecoder):
        wic_decoder = decoder

if not wic_decoder:
    raise Exception("Cannot use DirectWrite without a WIC Decoder")


class DirectWriteGlyphRenderer(base.GlyphRenderer):
    antialias_mode = D2D1_TEXT_ANTIALIAS_MODE_DEFAULT
    draw_options = D2D1_DRAW_TEXT_OPTIONS_ENABLE_COLOR_FONT

    def __init__(self, font):
        self._render_target = None
        self._bitmap = None
        self._brush = None
        self._bitmap_dimensions = (0, 0)
        super(DirectWriteGlyphRenderer, self).__init__(font)
        self.font = font

        self._analyzer = IDWriteTextAnalyzer()
        self.font._write_factory.CreateTextAnalyzer(byref(self._analyzer))

        self._text_analysis = TextAnalysis()

    def render_to_image(self, text, width, height):
        """This process takes Pyglet out of the equation and uses only DirectWrite to shape and render text.
        This may allows more accurate fonts (bidi, rtl, etc) in very special circumstances."""
        text_buffer = create_unicode_buffer(text)

        text_layout = IDWriteTextLayout()
        self.font._write_factory.CreateTextLayout(
            text_buffer,
            len(text_buffer),
            self.font._text_format,
            width,  # Doesn't affect bitmap size.
            height,
            byref(text_layout)
        )

        layout_metrics = DWRITE_TEXT_METRICS()
        text_layout.GetMetrics(byref(layout_metrics))

        width, height = int(math.ceil(layout_metrics.width)), int(math.ceil(layout_metrics.height))

        bitmap = IWICBitmap()
        wic_decoder._factory.CreateBitmap(width, height,
                                          GUID_WICPixelFormat32bppPBGRA,
                                          WICBitmapCacheOnDemand,
                                          byref(bitmap))

        rt = ID2D1RenderTarget()
        d2d_factory.CreateWicBitmapRenderTarget(bitmap, default_target_properties, byref(rt))

        # Font aliasing rendering quality.
        rt.SetTextAntialiasMode(self.antialias_mode)

        if not self._brush:
            self._brush = ID2D1SolidColorBrush()

        rt.CreateSolidColorBrush(white, None, byref(self._brush))

        # This offsets the characters if needed.
        point = D2D_POINT_2F(0, 0)

        rt.BeginDraw()

        rt.Clear(transparent)

        rt.DrawTextLayout(point,
                          text_layout,
                          self._brush,
                          self.draw_options)

        rt.EndDraw(None, None)

        rt.Release()

        image_data = wic_decoder.get_image(bitmap)

        return image_data

    def get_string_info(self, text):

        """Converts a string of text into a list of indices and advances used for shaping."""
        text_length = len(text.encode('utf-16-le')) // 2

        # Unicode buffer splits each two byte chars into separate indices.
        text_buffer = create_unicode_buffer(text, text_length)

        # Analyze the text.
        # noinspection PyTypeChecker
        self._text_analysis.GenerateResults(self._analyzer, text_buffer, len(text_buffer))

        # Formula for text buffer size from Microsoft.
        max_glyph_size = int(3 * text_length / 2 + 16)

        length = text_length
        clusters = (UINT16 * length)()
        text_props = (DWRITE_SHAPING_TEXT_PROPERTIES * length)()
        indices = (UINT16 * max_glyph_size)()
        glyph_props = (DWRITE_SHAPING_GLYPH_PROPERTIES * max_glyph_size)()
        actual_count = UINT32()

        self._analyzer.GetGlyphs(text_buffer,
                                 length,
                                 self.font.font_face,
                                 False,  # sideways
                                 False,  # righttoleft
                                 self._text_analysis._script,  # scriptAnalysis
                                 None,  # localName
                                 None,  # numberSub
                                 None,  # typo features
                                 None,  # feature range length
                                 0,  # feature range
                                 max_glyph_size,  # max glyph size
                                 clusters,  # cluster map
                                 text_props,  # text props
                                 indices,  # glyph indices
                                 glyph_props,  # glyph pops
                                 byref(actual_count)  # glyph count
                                 )

        advances = (FLOAT * length)()
        offsets = (DWRITE_GLYPH_OFFSET * length)()
        self._analyzer.GetGlyphPlacements(text_buffer,
                                          clusters,
                                          text_props,
                                          text_length,
                                          indices,
                                          glyph_props,
                                          actual_count,
                                          self.font.font_face,
                                          self.font._font_metrics.designUnitsPerEm,
                                          False, False,
                                          self._text_analysis._script,
                                          self.font.locale,
                                          None,
                                          None,
                                          0,
                                          advances,
                                          offsets)

        return text_buffer, actual_count.value, indices, advances, offsets, clusters

    def get_glyph_metrics(self, font_face, indices, count):
        """Returns a list of tuples with the following metrics per indice:
            (glyph width, glyph height, lsb, advanceWidth)
        """
        glyph_metrics = (DWRITE_GLYPH_METRICS * count)()
        font_face.GetDesignGlyphMetrics(indices, count, glyph_metrics, False)

        metrics_out = []
        i = 0
        for metric in glyph_metrics:
            glyph_width = (metric.advanceWidth - metric.leftSideBearing - metric.rightSideBearing)

            # width must have a minimum of 1. For example, spaces are actually 0 width, still need glyph bitmap size.
            if glyph_width == 0:
                glyph_width = 1

            glyph_height = (metric.advanceHeight - metric.topSideBearing - metric.bottomSideBearing)

            lsb = metric.leftSideBearing

            bsb = metric.bottomSideBearing

            advance_width = metric.advanceWidth

            metrics_out.append((glyph_width, glyph_height, lsb, advance_width, bsb))
            i += 1

        return metrics_out

    def _get_single_glyph_run(self, font_face, size, indices, advances, offsets, sideways, bidi):
        run = DWRITE_GLYPH_RUN(
            font_face,
            size,
            1,
            indices,
            advances,
            offsets,
            sideways,
            bidi
        )
        return run

    def render_single_glyph(self, font_face, indice, advance, offset, metrics):
        """Renders a single glyph using D2D DrawGlyphRun"""
        glyph_width, glyph_height, lsb, font_advance, bsb = metrics  # We use a shaped advance instead of the fonts.

        # Slicing an array turns it into a python object. Maybe a better way to keep it a ctypes value?
        new_indice = (UINT16 * 1)(indice)
        new_advance = (FLOAT * 1)(advance)

        run = self._get_single_glyph_run(font_face,
                                         self.font._real_size,
                                         new_indice,  # indice,
                                         new_advance,  # advance,
                                         pointer(offset),  # offset,
                                         False,
                                         False)

        render_width = int(math.ceil((glyph_width) * self.font.font_scale_ratio))
        render_offset_x = int(math.floor(abs(lsb * self.font.font_scale_ratio)))
        if lsb < 0:
            # Negative LSB: we shift the layout rect to the right
            # Otherwise we will cut the left part of the glyph
            render_offset_x = -(render_offset_x)

        # Create new bitmap.
        # TODO: We can probably adjust bitmap/baseline to reduce the whitespace and save a lot of texture space.
        # Note: Floating point precision makes this a giant headache, will need to be solved for this approach.
        self._create_bitmap(render_width + 1,  # Add 1, sometimes AA can add an extra pixel or so.
                            int(math.ceil(self.font.max_glyph_height)))

        # Glyphs are drawn at the baseline, and with LSB, so we need to offset it based on top left position.
        # Offsets are actually based on pixels somehow???
        baseline_offset = D2D_POINT_2F(-render_offset_x - offset.advanceOffset,
                                       self.font.ascent + offset.ascenderOffset)

        self._render_target.BeginDraw()

        self._render_target.Clear(transparent)

        self._render_target.DrawGlyphRun(baseline_offset,
                                         run,
                                         self._brush,
                                         DWRITE_MEASURING_MODE_NATURAL)

        self._render_target.EndDraw(None, None)
        image = wic_decoder.get_image(self._bitmap)

        glyph = self.font.create_glyph(image)

        glyph.set_bearings(self.font.descent, render_offset_x,
                           advance * self.font.font_scale_ratio,
                           offset.advanceOffset * self.font.font_scale_ratio,
                           offset.ascenderOffset * self.font.font_scale_ratio)

        return glyph

    def render_using_layout(self, text):
        """This will render text given the built in DirectWrite layout. This process allows us to take
        advantage of color glyphs and fallback handling that is built into DirectWrite.
        This can also handle shaping and many other features if you want to render directly to a texture."""
        text_layout = self.font.create_text_layout(text)

        layout_metrics = DWRITE_TEXT_METRICS()
        text_layout.GetMetrics(byref(layout_metrics))

        self._create_bitmap(int(math.ceil(layout_metrics.width)),
                            int(math.ceil(layout_metrics.height)))

        # This offsets the characters if needed.
        point = D2D_POINT_2F(0, 0)

        self._render_target.BeginDraw()

        self._render_target.Clear(transparent)

        self._render_target.DrawTextLayout(point,
                                           text_layout,
                                           self._brush,
                                           self.draw_options)

        self._render_target.EndDraw(None, None)

        image = wic_decoder.get_image(self._bitmap)

        glyph = self.font.create_glyph(image)
        glyph.set_bearings(self.font.descent, 0, int(math.ceil(layout_metrics.width)))
        return glyph

    def _create_bitmap(self, width, height):
        """Creates a bitmap using Direct2D and WIC."""
        # Create a new bitmap, try to re-use the bitmap as much as we can to minimize creations.
        if self._bitmap_dimensions[0] != width or self._bitmap_dimensions[1] != height:
            # If dimensions aren't the same, release bitmap to create new ones.
            if self._bitmap:
                self._bitmap.Release()

            self._bitmap = IWICBitmap()
            wic_decoder._factory.CreateBitmap(width, height,
                                              GUID_WICPixelFormat32bppPBGRA,
                                              WICBitmapCacheOnDemand,
                                              byref(self._bitmap))

            self._render_target = ID2D1RenderTarget()
            d2d_factory.CreateWicBitmapRenderTarget(self._bitmap, default_target_properties, byref(self._render_target))

            # Font aliasing rendering quality.
            self._render_target.SetTextAntialiasMode(self.antialias_mode)

            if not self._brush:
                self._brush = ID2D1SolidColorBrush()
                self._render_target.CreateSolidColorBrush(white, None, byref(self._brush))


class Win32DirectWriteFont(base.Font):
    # To load fonts from files, we need to produce a custom collection.
    _custom_collection = None

    # Shared loader values
    _write_factory = None  # Factory required to run any DirectWrite interfaces.
    _font_loader = None

    # Windows 10 loader values.
    _font_builder = None
    _font_set = None

    # Legacy loader values
    _font_collection_loader = None
    _font_cache = []
    _font_loader_key = None

    _default_name = 'Segoe UI'  # Default font for Win7/10.

    _glyph_renderer = None

    glyph_renderer_class = DirectWriteGlyphRenderer
    texture_internalformat = pyglet.gl.GL_RGBA

    def __init__(self, name, size, bold=False, italic=False, stretch=False, dpi=None, locale=None):
        self._advance_cache = {}  # Stores glyph's by the indice and advance.

        super(Win32DirectWriteFont, self).__init__()

        if not name:
            name = self._default_name

        self._font_index, self._collection = self.get_collection(name)
        assert self._collection is not None, "Font: {} not found in loaded or system font collection.".format(name)

        self._name = name
        self.bold = bold
        self.size = size
        self.italic = italic
        self.stretch = stretch
        self.dpi = dpi
        self.locale = locale

        if self.locale is None:
            self.locale = ""
            self.rtl = False  # Right to left should be handled by pyglet?
            # TODO: Use system locale string?

        if self.dpi is None:
            self.dpi = 96

        # From DPI to DIP (Device Independent Pixels) which is what the fonts rely on.
        self._real_size = (self.size * self.dpi) // 72

        if self.bold:
            if type(self.bold) is str:
                self._weight = name_to_weight[self.bold]
            else:
                self._weight = DWRITE_FONT_WEIGHT_BOLD
        else:
            self._weight = DWRITE_FONT_WEIGHT_NORMAL

        if self.italic:
            if type(self.italic) is str:
                self._style = name_to_style[self.italic]
            else:
                self._style = DWRITE_FONT_STYLE_ITALIC
        else:
            self._style = DWRITE_FONT_STYLE_NORMAL

        if self.stretch:
            if type(self.stretch) is str:
                self._stretch = name_to_stretch[self.stretch]
            else:
                self._stretch = DWRITE_FONT_STRETCH_EXPANDED
        else:
            self._stretch = DWRITE_FONT_STRETCH_NORMAL

        # Create the text format this font will use permanently.
        # Could technically be recreated, but will keep to be inline with other font objects.
        self._text_format = IDWriteTextFormat()
        self._write_factory.CreateTextFormat(self._name,
                                             self._collection,
                                             self._weight,
                                             self._style,
                                             self._stretch,
                                             self._real_size,
                                             create_unicode_buffer(self.locale),
                                             byref(self._text_format))

        # All this work just to get a font face and it's metrics!
        font_family = IDWriteFontFamily1()
        self._collection.GetFontFamily(self._font_index, byref(font_family))

        write_font = IDWriteFont()
        font_family.GetFirstMatchingFont(self._weight,
                                         self._stretch,
                                         self._style,
                                         byref(write_font))

        font_face = IDWriteFontFace()
        write_font.CreateFontFace(byref(font_face))

        self.font_face = IDWriteFontFace1()
        font_face.QueryInterface(IID_IDWriteFontFace1, byref(self.font_face))

        self._font_metrics = DWRITE_FONT_METRICS()
        self.font_face.GetMetrics(byref(self._font_metrics))

        self.font_scale_ratio = (self._real_size / self._font_metrics.designUnitsPerEm)

        self.ascent = self._font_metrics.ascent * self.font_scale_ratio
        self.descent = self._font_metrics.descent * self.font_scale_ratio

        self.max_glyph_height = (self._font_metrics.ascent + self._font_metrics.descent) * self.font_scale_ratio
        self.line_gap = self._font_metrics.lineGap * self.font_scale_ratio

    @property
    def name(self):
        return self._name

    def render_to_image(self, text, width=10000, height=80):
        """This process takes Pyglet out of the equation and uses only DirectWrite to shape and render text.
        This may allow more accurate fonts (bidi, rtl, etc) in very special circumstances at the cost of
        additional texture space.

        :Parameters:
            `text` : str
                String of text to render.

        :rtype: `ImageData`
        :return: An image of the text.
        """
        if not self._glyph_renderer:
            self._glyph_renderer = self.glyph_renderer_class(self)

        return self._glyph_renderer.render_to_image(text, width, height)

    def copy_glyph(self, glyph, advance, offset):
        """This takes the existing glyph texture and puts it into a new Glyph with a new advance.
        Texture memory is shared between both glyphs."""
        new_glyph = base.Glyph(glyph.x, glyph.y, glyph.z, glyph.width, glyph.height, glyph.owner)
        new_glyph.set_bearings(glyph.baseline,
                               glyph.lsb,
                               advance * self.font_scale_ratio,
                               offset.advanceOffset * self.font_scale_ratio,
                               offset.ascenderOffset * self.font_scale_ratio)
        return new_glyph

    def get_glyphs(self, text):
        if not self._glyph_renderer:
            self._glyph_renderer = self.glyph_renderer_class(self)

        text_buffer, actual_count, indices, advances, offsets, clusters = self._glyph_renderer.get_string_info(text)

        metrics = self._glyph_renderer.get_glyph_metrics(self.font_face, indices, actual_count)

        glyphs = []
        for i in range(actual_count):
            indice = indices[i]
            if indice == 0:
                # If an indice is 0, it will return no glyph. In this case we attempt to render leveraging
                # the built in text layout from MS. Which depending on version can use fallback fonts and other tricks
                # to possibly get something of use.
                formatted_clusters = clusters[:]

                # Some glyphs can be more than 1 char. We use the clusters to determine how many of an index exist.
                text_length = formatted_clusters.count(i)

                # Amount of glyphs don't always match 1:1 with text as some can be substituted or omitted. Get
                # actual text buffer index.
                text_index = formatted_clusters.index(i)

                # Get actual text based on the index and length.
                actual_text = text_buffer[text_index:text_index + text_length]

                # Since we can't store as indice 0 without overriding, we have to store as text
                if actual_text not in self.glyphs:
                    glyph = self._glyph_renderer.render_using_layout(text_buffer[text_index:text_index + text_length])
                    self.glyphs[actual_text] = glyph

                glyphs.append(self.glyphs[actual_text])
            else:
                # Glyphs can vary depending on shaping. We will cache it by indice, advance, and offset.
                # Possible to just cache without offset and set them each time. This may be faster?
                if indice in self.glyphs:
                    advance_key = (indice, advances[i], offsets[i].advanceOffset, offsets[i].ascenderOffset)
                    if advance_key in self._advance_cache:
                        glyph = self._advance_cache[advance_key]
                    else:
                        glyph = self.copy_glyph(self.glyphs[indice], advances[i], offsets[i])
                        self._advance_cache[advance_key] = glyph
                else:
                    glyph = self._glyph_renderer.render_single_glyph(self.font_face, indice, advances[i], offsets[i],
                                                                     metrics[i])
                    self.glyphs[indice] = glyph
                    self._advance_cache[(indice, advances[i], offsets[i].advanceOffset, offsets[i].ascenderOffset)] = glyph

                glyphs.append(glyph)

        return glyphs

    def create_text_layout(self, text):
        text_buffer = create_unicode_buffer(text)

        text_layout = IDWriteTextLayout()
        hr = self._write_factory.CreateTextLayout(text_buffer,
                                                  len(text_buffer),
                                                  self._text_format,
                                                  10000,  # Doesn't affect bitmap size.
                                                  80,
                                                  byref(text_layout)
                                                  )

        return text_layout

    @classmethod
    def _initialize_direct_write(cls):
        """ All direct write fonts needs factory access as well as the loaders."""
        if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
            cls._write_factory = IDWriteFactory5()
            DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, IID_IDWriteFactory5, byref(cls._write_factory))
        else:
            # Windows 7 and 8 we need to create our own font loader, collection, enumerator, file streamer... Sigh.
            cls._write_factory = IDWriteFactory()
            DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, IID_IDWriteFactory, byref(cls._write_factory))

    @classmethod
    def _initialize_custom_loaders(cls):
        """Initialize the loaders needed to load custom fonts."""
        if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
            # Windows 10 finally has a built in loader that can take data and make a font out of it w/ COMs.
            cls._font_loader = IDWriteInMemoryFontFileLoader()
            cls._write_factory.CreateInMemoryFontFileLoader(byref(cls._font_loader))
            cls._write_factory.RegisterFontFileLoader(cls._font_loader)

            # Used for grouping fonts together.
            cls._font_builder = IDWriteFontSetBuilder1()
            cls._write_factory.CreateFontSetBuilder1(byref(cls._font_builder))
        else:
            cls._font_loader = LegacyFontFileLoader()

            # Note: RegisterFontLoader takes a pointer. However, for legacy we implement our own callback interface.
            # Therefore we need to pass to the actual pointer directly.
            cls._write_factory.RegisterFontFileLoader(cls._font_loader.pointers[IDWriteFontFileLoader])

            cls._font_collection_loader = LegacyCollectionLoader(cls._write_factory, cls._font_loader)
            cls._write_factory.RegisterFontCollectionLoader(cls._font_collection_loader)

            cls._font_loader_key = cast(create_unicode_buffer("legacy_font_loader"), c_void_p)

    @classmethod
    def add_font_data(cls, data):
        if not cls._write_factory:
            cls._initialize_direct_write()

        if not cls._font_loader:
            cls._initialize_custom_loaders()

        if WINDOWS_10_CREATORS_UPDATE_OR_GREATER:
            font_file = IDWriteFontFile()
            hr = cls._font_loader.CreateInMemoryFontFileReference(cls._write_factory,
                                                                  data,
                                                                  len(data),
                                                                  None,
                                                                  byref(font_file))

            hr = cls._font_builder.AddFontFile(font_file)
            if hr != 0:
                raise Exception("This font file data is not not a font or unsupported.")

            # We have to rebuild collection everytime we add a font.
            # No way to add fonts to the collection once the FontSet and Collection are created.
            # Release old one and renew.
            if cls._custom_collection:
                cls._font_set.Release()
                cls._custom_collection.Release()

            cls._font_set = IDWriteFontSet()
            cls._font_builder.CreateFontSet(byref(cls._font_set))

            cls._custom_collection = IDWriteFontCollection1()
            cls._write_factory.CreateFontCollectionFromFontSet(cls._font_set, byref(cls._custom_collection))
        else:
            cls._font_cache.append(data)

            # If a collection exists, we need to completely remake the collection, delete everything and start over.
            if cls._custom_collection:
                cls._custom_collection = None

                cls._write_factory.UnregisterFontCollectionLoader(cls._font_collection_loader)
                cls._write_factory.UnregisterFontFileLoader(cls._font_loader)

                cls._font_loader = LegacyFontFileLoader()
                cls._font_collection_loader = LegacyCollectionLoader(cls._write_factory, cls._font_loader)

                cls._write_factory.RegisterFontCollectionLoader(cls._font_collection_loader)
                cls._write_factory.RegisterFontFileLoader(cls._font_loader.pointers[IDWriteFontFileLoader])

            cls._font_collection_loader.AddFontData(cls._font_cache)

            cls._custom_collection = IDWriteFontCollection()

            cls._write_factory.CreateCustomFontCollection(cls._font_collection_loader,
                                                          cls._font_loader_key,
                                                          sizeof(cls._font_loader_key),
                                                          byref(cls._custom_collection))

    @classmethod
    def get_collection(cls, font_name):
        """Returns which collection this font belongs to (system or custom collection), as well as it's index in the
        collection."""
        if not cls._write_factory:
            cls._initialize_direct_write()

        """Returns a collection the font_name belongs to."""
        font_index = UINT()
        font_exists = BOOL()

        # Check custom loaded font collections.
        if cls._custom_collection:
            cls._custom_collection.FindFamilyName(create_unicode_buffer(font_name),
                                                  byref(font_index),
                                                  byref(font_exists))

            if font_exists.value:
                return font_index.value, cls._custom_collection

        # Check if font is in the system collection.
        # Do not cache these values permanently as system font collection can be updated during runtime.
        if not font_exists.value:
            sys_collection = IDWriteFontCollection()
            cls._write_factory.GetSystemFontCollection(byref(sys_collection), 1)
            sys_collection.FindFamilyName(create_unicode_buffer(font_name),
                                          byref(font_index),
                                          byref(font_exists))

            if font_exists.value:
                return font_index.value, sys_collection

        # Font does not exist in either custom or system.
        return None, None

    @classmethod
    def have_font(cls, name):
        if cls.get_collection(name)[0] is not None:
            return True

        return False

    @classmethod
    def get_font_face(cls, name):
        # Check custom collection.
        collection = None
        font_index = UINT()
        font_exists = BOOL()

        # Check custom collection.
        if cls._custom_collection:
            cls._custom_collection.FindFamilyName(create_unicode_buffer(name),
                                                  byref(font_index),
                                                  byref(font_exists))

            collection = cls._custom_collection

        if font_exists.value == 0:
            sys_collection = IDWriteFontCollection()
            cls._write_factory.GetSystemFontCollection(byref(sys_collection), 1)
            sys_collection.FindFamilyName(create_unicode_buffer(name),
                                          byref(font_index),
                                          byref(font_exists))

            collection = sys_collection

        if font_exists:
            font_family = IDWriteFontFamily()
            collection.GetFontFamily(font_index, byref(font_family))

            write_font = IDWriteFont()
            font_family.GetFirstMatchingFont(DWRITE_FONT_WEIGHT_NORMAL,
                                             DWRITE_FONT_STRETCH_NORMAL,
                                             DWRITE_FONT_STYLE_NORMAL,
                                             byref(write_font))

            font_face = IDWriteFontFace1()
            write_font.CreateFontFace(byref(font_face))

            return font_face

        return None


d2d_factory = ID2D1Factory()
hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, IID_ID2D1Factory, None, byref(d2d_factory))

WICBitmapCreateCacheOption = UINT
WICBitmapNoCache = 0
WICBitmapCacheOnDemand = 0x1
WICBitmapCacheOnLoad = 0x2

transparent = D2D1_COLOR_F(0.0, 0.0, 0.0, 0.0)
white = D2D1_COLOR_F(1.0, 1.0, 1.0, 1.0)
