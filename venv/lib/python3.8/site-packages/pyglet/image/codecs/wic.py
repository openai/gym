import warnings
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet import com
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32

CLSID_WICImagingFactory1 = com.GUID(0xcacaf262, 0x9370, 0x4615, 0xa1, 0x3b, 0x9f, 0x55, 0x39, 0xda, 0x4c, 0xa)
CLSID_WICImagingFactory2 = com.GUID(0x317d06e8, 0x5f24, 0x433d, 0xbd, 0xf7, 0x79, 0xce, 0x68, 0xd8, 0xab, 0xc2)

# This is available with Windows 7 with a Platform Update, but unable to detect as it wasn't a version change to the OS,
# but a KB update. Available in atleast 8+.
if WINDOWS_8_OR_GREATER:
    CLSID_WICImagingFactory = CLSID_WICImagingFactory2
else:
    CLSID_WICImagingFactory = CLSID_WICImagingFactory1

WICBitmapCreateCacheOption = UINT
WICBitmapNoCache = 0
WICBitmapCacheOnDemand = 0x1
WICBitmapCacheOnLoad = 0x2
WICBITMAPCREATECACHEOPTION_FORCE_DWORD = 0x7fffffff

WICBitmapPaletteType = UINT
WICBitmapPaletteTypeCustom = 0

WICBitmapTransformOptions = UINT
WICBitmapTransformRotate0 = 0
WICBitmapTransformRotate90 = 0x1
WICBitmapTransformRotate180 = 0x2
WICBitmapTransformRotate270 = 0x3
WICBitmapTransformFlipHorizontal = 0x8
WICBitmapTransformFlipVertical = 0x10

WICBitmapDitherType = UINT
WICBitmapDitherTypeNone = 0
WICBitmapDitherTypeSolid = 0
WICBitmapDitherTypeOrdered4x4 = 0x1
WICBitmapDitherTypeOrdered8x8 = 0x2
WICBitmapDitherTypeOrdered16x16 = 0x3
WICBitmapDitherTypeSpiral4x4 = 0x4
WICBitmapDitherTypeSpiral8x8 = 0x5
WICBitmapDitherTypeDualSpiral4x4 = 0x6
WICBitmapDitherTypeDualSpiral8x8 = 0x7
WICBitmapDitherTypeErrorDiffusion = 0x8
WICBITMAPDITHERTYPE_FORCE_DWORD = 0x7fffffff
WICBITMAPTRANSFORMOPTIONS_FORCE_DWORD = 0x7fffffff


WICDecodeOptions = UINT
WICDecodeMetadataCacheOnDemand = 0
WICDecodeMetadataCacheOnLoad = 0x1
WICMETADATACACHEOPTION_FORCE_DWORD = 0x7fffffff

# Different pixel formats.
REFWICPixelFormatGUID = com.GUID
GUID_WICPixelFormat1bppIndexed = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x01)
GUID_WICPixelFormat2bppIndexed = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x02)
GUID_WICPixelFormat4bppIndexed = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x03)
GUID_WICPixelFormat8bppIndexed = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x04)
GUID_WICPixelFormatBlackWhite = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x05)
GUID_WICPixelFormat2bppGray = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x06)
GUID_WICPixelFormat4bppGray = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x07)
GUID_WICPixelFormat8bppGray = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x08)
GUID_WICPixelFormat8bppAlpha = com.GUID(0xe6cd0116, 0xeeba, 0x4161, 0xaa, 0x85, 0x27, 0xdd, 0x9f, 0xb3, 0xa8, 0x95)
GUID_WICPixelFormat16bppBGR555 = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x09)
GUID_WICPixelFormat16bppBGR565 = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x0a)
GUID_WICPixelFormat16bppBGRA5551 = com.GUID(0x05ec7c2b, 0xf1e6, 0x4961, 0xad, 0x46, 0xe1, 0xcc, 0x81, 0x0a, 0x87, 0xd2)
GUID_WICPixelFormat16bppGray = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x0b)
GUID_WICPixelFormat24bppBGR = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x0c)
GUID_WICPixelFormat24bppRGB = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x0d)
GUID_WICPixelFormat32bppBGR = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x0e)
GUID_WICPixelFormat32bppBGRA = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x0f)
GUID_WICPixelFormat32bppPBGRA = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x10)
GUID_WICPixelFormat32bppRGB = com.GUID(0xd98c6b95, 0x3efe, 0x47d6, 0xbb, 0x25, 0xeb, 0x17, 0x48, 0xab, 0x0c, 0xf1)  # 7 platform update?
GUID_WICPixelFormat32bppRGBA = com.GUID(0xf5c7ad2d, 0x6a8d, 0x43dd, 0xa7, 0xa8, 0xa2, 0x99, 0x35, 0x26, 0x1a, 0xe9)
GUID_WICPixelFormat32bppPRGBA = com.GUID(0x3cc4a650, 0xa527, 0x4d37, 0xa9, 0x16, 0x31, 0x42, 0xc7, 0xeb, 0xed, 0xba)
GUID_WICPixelFormat48bppRGB = com.GUID(0x6fddc324, 0x4e03, 0x4bfe, 0xb1, 0x85, 0x3d, 0x77, 0x76, 0x8d, 0xc9, 0x15)
GUID_WICPixelFormat48bppBGR = com.GUID(0xe605a384, 0xb468, 0x46ce, 0xbb, 0x2e, 0x36, 0xf1, 0x80, 0xe6, 0x43, 0x13)


class IWICComponentInfo(com.pIUnknown):
    _methods_ = [
        ('GetComponentType',
         com.STDMETHOD()),
        ('GetCLSID',
         com.STDMETHOD()),
        ('GetSigningStatus',
         com.STDMETHOD()),
        ('GetAuthor',
         com.STDMETHOD()),
        ('GetVendorGUID',
         com.STDMETHOD()),
        ('GetVersion',
         com.STDMETHOD()),
        ('GetSpecVersion',
         com.STDMETHOD()),
        ('GetFriendlyName',
         com.STDMETHOD())
    ]


class IWICPixelFormatInfo(IWICComponentInfo, com.pIUnknown):
    _methods_ = [
        ('GetFormatGUID',
         com.STDMETHOD(POINTER(com.GUID))),
        ('GetColorContext',
         com.STDMETHOD()),
        ('GetBitsPerPixel',
         com.STDMETHOD(POINTER(UINT))),
        ('GetChannelCount',
         com.STDMETHOD(POINTER(UINT))),
        ('GetChannelMask',
         com.STDMETHOD())
    ]


class IWICBitmapSource(com.pIUnknown):
    _methods_ = [
        ('GetSize',
         com.STDMETHOD(POINTER(UINT), POINTER(UINT))),
        ('GetPixelFormat',
         com.STDMETHOD(POINTER(REFWICPixelFormatGUID))),
        ('GetResolution',
         com.STDMETHOD(POINTER(DOUBLE), POINTER(DOUBLE))),
        ('CopyPalette',
         com.STDMETHOD()),
        ('CopyPixels',
         com.STDMETHOD(c_void_p, UINT, UINT, c_void_p)),
    ]


class IWICFormatConverter(IWICBitmapSource, com.pIUnknown):
    _methods_ = [
        ('Initialize',
         com.STDMETHOD(IWICBitmapSource, POINTER(REFWICPixelFormatGUID), WICBitmapDitherType, c_void_p, DOUBLE, WICBitmapPaletteType)),
        ('CanConvert',
         com.STDMETHOD(POINTER(REFWICPixelFormatGUID), POINTER(REFWICPixelFormatGUID), POINTER(BOOL))),
    ]


class IWICMetadataQueryReader(com.pIUnknown):
    _methods_ = [
        ('GetContainerFormat',
         com.STDMETHOD()),
        ('GetLocation',
         com.STDMETHOD()),
        ('GetMetadataByName',
         com.STDMETHOD(LPCWSTR, c_void_p)),
        ('GetEnumerator',
         com.STDMETHOD()),
    ]


class IWICBitmapFrameDecode(IWICBitmapSource, com.pIUnknown):
    _methods_ = [
        ('GetMetadataQueryReader',
         com.STDMETHOD(POINTER(IWICMetadataQueryReader))),
        ('GetColorContexts',
         com.STDMETHOD()),
        ('GetThumbnail',
         com.STDMETHOD(POINTER(IWICBitmapSource))),
    ]


class IWICBitmapFlipRotator(IWICBitmapSource, com.pIUnknown):
    _methods_ = [
        ('Initialize',
         com.STDMETHOD(IWICBitmapSource, WICBitmapTransformOptions)),
    ]


class IWICBitmap(IWICBitmapSource, com.pIUnknown):
    _methods_ = [
        ('Lock',
         com.STDMETHOD()),
        ('SetPalette',
         com.STDMETHOD()),
        ('SetResolution',
         com.STDMETHOD())
    ]


class IWICBitmapDecoder(com.pIUnknown):
    _methods_ = [
        ('QueryCapability',
         com.STDMETHOD()),
        ('Initialize',
         com.STDMETHOD()),
        ('GetContainerFormat',
         com.STDMETHOD()),
        ('GetDecoderInfo',
         com.STDMETHOD()),
        ('CopyPalette',
         com.STDMETHOD()),
        ('GetMetadataQueryReader',
         com.STDMETHOD(POINTER(IWICMetadataQueryReader))),
        ('GetPreview',
         com.STDMETHOD()),
        ('GetColorContexts',
         com.STDMETHOD()),
        ('GetThumbnail',
         com.STDMETHOD()),
        ('GetFrameCount',
         com.STDMETHOD(POINTER(UINT))),
        ('GetFrame',
         com.STDMETHOD(UINT, POINTER(IWICBitmapFrameDecode))),
    ]


IID_IWICImagingFactory1 = com.GUID(0xec5ec8a9, 0xc395, 0x4314, 0x9c, 0x77, 0x54, 0xd7, 0xa9, 0x35, 0xff, 0x70)
IID_IWICImagingFactory2 = com.GUID(0x7B816B45, 0x1996, 0x4476, 0xB1, 0x32, 0xDE, 0x9E, 0x24, 0x7C, 0x8A, 0xF0)

if WINDOWS_8_OR_GREATER:
    IID_IWICImagingFactory = IID_IWICImagingFactory2
else:
    IID_IWICImagingFactory = IID_IWICImagingFactory1

IID_IWICPixelFormatInfo = com.GUID(0xE8EDA601, 0x3D48, 0x431a, 0xAB, 0x44, 0x69, 0x05, 0x9B, 0xE8, 0x8B, 0xBE)


class IWICImagingFactory(com.pIUnknown):
    _methods_ = [
        ('CreateDecoderFromFilename',
         com.STDMETHOD(LPCWSTR, com.GUID, DWORD, WICDecodeOptions, POINTER(IWICBitmapDecoder))),
        ('CreateDecoderFromStream',
         com.STDMETHOD(com.pIUnknown, c_void_p, WICDecodeOptions, POINTER(IWICBitmapDecoder))),
        ('CreateDecoderFromFileHandle',
         com.STDMETHOD()),
        ('CreateComponentInfo',
         com.STDMETHOD(com.GUID, POINTER(IWICComponentInfo))),
        ('CreateDecoder',
         com.STDMETHOD()),
        ('CreateEncoder',
         com.STDMETHOD()),
        ('CreatePalette',
         com.STDMETHOD()),
        ('CreateFormatConverter',
         com.STDMETHOD(POINTER(IWICFormatConverter))),
        ('CreateBitmapScaler',
         com.STDMETHOD()),
        ('CreateBitmapClipper',
         com.STDMETHOD()),
        ('CreateBitmapFlipRotator',
         com.STDMETHOD(POINTER(IWICBitmapFlipRotator))),
        ('CreateStream',
         com.STDMETHOD()),
        ('CreateColorContext',
         com.STDMETHOD()),
        ('CreateColorTransformer',
         com.STDMETHOD()),
        ('CreateBitmap',
         com.STDMETHOD(UINT, UINT, POINTER(REFWICPixelFormatGUID), WICBitmapCreateCacheOption, POINTER(IWICBitmap))),
        ('CreateBitmapFromSource',
         com.STDMETHOD()),
        ('CreateBitmapFromSourceRect',
         com.STDMETHOD()),
        ('CreateBitmapFromMemory',
         com.STDMETHOD()),
        ('CreateBitmapFromHBITMAP',
         com.STDMETHOD()),
        ('CreateBitmapFromHICON',
         com.STDMETHOD()),
        ('CreateComponentEnumerator',
         com.STDMETHOD()),
        ('CreateFastMetadataEncoderFromDecoder',
         com.STDMETHOD()),
        ('CreateFastMetadataEncoderFromFrameDecode',
         com.STDMETHOD()),
        ('CreateQueryWriter',
         com.STDMETHOD()),
        ('CreateQueryWriterFromReader',
         com.STDMETHOD())
    ]


class WICDecoder(ImageDecoder):
    """Windows Imaging Component.
    This decoder is a replacement for GDI and GDI+ starting with Windows 7 with more features up to Windows 10."""
    def __init__(self):
        super(ImageDecoder, self).__init__()
        self._factory = IWICImagingFactory()

        try:
            ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        except OSError as err:
            warnings.warn(str(err))

        ole32.CoCreateInstance(CLSID_WICImagingFactory,
                               None,
                               CLSCTX_INPROC_SERVER,
                               IID_IWICImagingFactory,
                               byref(self._factory))

    def get_file_extensions(self):
        return ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ico', '.jxr', '.hdp', '.wdp']

    def _load_bitmap_decoder(self, file, filename):
        data = file.read()

        # Create a HGLOBAL with image data
        hglob = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        ptr = kernel32.GlobalLock(hglob)
        memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(hglob)

        # Create IStream for the HGLOBAL
        stream = com.pIUnknown()
        ole32.CreateStreamOnHGlobal(hglob, True, byref(stream))

        # Load image from stream
        decoder = IWICBitmapDecoder()
        status = self._factory.CreateDecoderFromStream(stream, None, WICDecodeMetadataCacheOnDemand, byref(decoder))
        if status != 0:
            stream.Release()
            raise ImageDecodeException('WIC cannot load %r' % (filename or file))

        return decoder, stream

    def _get_bitmap_frame(self, bitmap_decoder, frame_index):
        bitmap = IWICBitmapFrameDecode()
        bitmap_decoder.GetFrame(frame_index, byref(bitmap))
        return bitmap

    def get_image(self, bitmap, target_fmt=GUID_WICPixelFormat32bppBGRA):
        """Get's image from bitmap, specifying target format, bitmap is released before returning."""
        width = UINT()
        height = UINT()

        bitmap.GetSize(byref(width), byref(height))

        width = int(width.value)
        height = int(height.value)

        # Get image pixel format
        pf = com.GUID(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        bitmap.GetPixelFormat(byref(pf))

        fmt = 'BGRA'
        # If target format is not what we want (32bit BGRA) convert it.
        if pf != target_fmt:
            converter = IWICFormatConverter()
            self._factory.CreateFormatConverter(byref(converter))

            conversion_possible = BOOL()
            converter.CanConvert(pf, target_fmt, byref(conversion_possible))

            # 99% of the time conversion will be possible to default.
            # However, we check to be safe and fallback to 24 bit BGR if not possible.
            if not conversion_possible:
                target_fmt = GUID_WICPixelFormat24bppBGR
                fmt = 'BGR'

            converter.Initialize(bitmap, target_fmt, WICBitmapDitherTypeNone, None, 0, WICBitmapPaletteTypeCustom)

            bitmap.Release()
            bitmap = converter

        # Most images are loaded with a negative pitch, which requires list comprehension to fix.
        # Create a flipped bitmap through the decoder rather through Python to increase performance.
        flipper = IWICBitmapFlipRotator()
        self._factory.CreateBitmapFlipRotator(byref(flipper))

        flipper.Initialize(bitmap, WICBitmapTransformFlipVertical)

        stride = len(fmt) * width
        buffer_size = stride * height

        buffer = (BYTE * buffer_size)()

        flipper.CopyPixels(None, stride, buffer_size, byref(buffer))

        flipper.Release()
        bitmap.Release()  # Can be converter.

        return ImageData(width, height, fmt, buffer)

    def _delete_bitmap_decoder(self, bitmap_decoder, stream):
        # Release decoder and stream
        bitmap_decoder.Release()
        stream.Release()

    def decode(self, file, filename):
        bitmap_decoder, stream = self._load_bitmap_decoder(file, filename)
        bitmap = self._get_bitmap_frame(bitmap_decoder, 0)
        image = self.get_image(bitmap)
        self._delete_bitmap_decoder(bitmap_decoder, stream)
        return image

    @staticmethod
    def get_property_value(reader, metadata_name):
        """
            Uses a metadata name and reader to return a single value. Can be used to get metadata from images.
            If failure, returns 0.
            Also handles cleanup of PROPVARIANT.
        """
        try:
            prop = PROPVARIANT()
            reader.GetMetadataByName(metadata_name, byref(prop))
            value = prop.llVal
            ole32.PropVariantClear(byref(prop))
        except OSError:
            value = 0

        return value


def get_decoders():
    return [WICDecoder()]


def get_encoders():
    return []
