dolphin_ini = """
[General]
LastFilename = SSBM.iso
ShowLag = False
ShowFrameCount = False
ISOPaths = 2
RecursiveISOPaths = False
NANDRootPath =
WirelessMac =
ISOPath0 =
ISOPath1 = ./
[Interface]
ConfirmStop = True
UsePanicHandlers = True
OnScreenDisplayMessages = True
HideCursor = False
AutoHideCursor = False
MainWindowPosX = 100
MainWindowPosY = 156
MainWindowWidth = 400
MainWindowHeight = 328
Language = 0
ShowToolbar = True
ShowStatusbar = True
ShowLogWindow = False
ShowLogConfigWindow = False
ExtendedFPSInfo = False
ThemeName40 = Clean
PauseOnFocusLost = False
[Display]
FullscreenResolution = Auto
Fullscreen = False
RenderToMain = False
RenderWindowXPos = 0
RenderWindowYPos = 0
RenderWindowWidth = 640
RenderWindowHeight = 528
RenderWindowAutoSize = False
KeepWindowOnTop = False
ProgressiveScan = False
PAL60 = True
DisableScreenSaver = True
ForceNTSCJ = False
[GameList]
ListDrives = False
ListWad = True
ListElfDol = True
ListWii = True
ListGC = True
ListJap = True
ListPal = True
ListUsa = True
ListAustralia = True
ListFrance = True
ListGermany = True
ListItaly = True
ListKorea = True
ListNetherlands = True
ListRussia = True
ListSpain = True
ListTaiwan = True
ListWorld = True
ListUnknown = True
ListSort = 3
ListSortSecondary = 0
ColorCompressed = True
ColumnPlatform = True
ColumnBanner = True
ColumnNotes = True
ColumnFileName = False
ColumnID = False
ColumnRegion = True
ColumnSize = True
ColumnState = True
[Core]
HLE_BS2 = True
TimingVariance = 40
CPUCore = 1
Fastmem = True
CPUThread = {cpu_thread}
DSPHLE = True
SkipIdle = True
SyncOnSkipIdle = True
SyncGPU = False
SyncGpuMaxDistance = 200000
SyncGpuMinDistance = -200000
SyncGpuOverclock = 1.00000000
FPRF = False
AccurateNaNs = False
DefaultISO =
DVDRoot =
Apploader =
EnableCheats = True
SelectedLanguage = 0
OverrideGCLang = False
DPL2Decoder = False
Latency = 2
MemcardAPath = {user}/GC/MemoryCardA.USA.raw
MemcardBPath = {user}/GC/MemoryCardB.USA.raw
AgpCartAPath =
AgpCartBPath =
SlotA = 255
SlotB = 255
SerialPort1 = 255
BBA_MAC =
SIDevice0 = 6
AdapterRumble0 = True
SimulateKonga0 = False
SIDevice1 = 6
AdapterRumble1 = True
SimulateKonga1 = False
SIDevice2 = 0
AdapterRumble2 = True
SimulateKonga2 = False
SIDevice3 = 0
AdapterRumble3 = True
SimulateKonga3 = False
WiiSDCard = False
WiiKeyboard = False
WiimoteContinuousScanning = False
WiimoteEnableSpeaker = False
RunCompareServer = False
RunCompareClient = False
EmulationSpeed = {speed}
FrameSkip = 0x00000000
Overclock = 1.00000000
OverclockEnable = False
GFXBackend = {gfx}
GPUDeterminismMode = auto
PerfMapDir =
[Movie]
PauseMovie = False
Author =
DumpFrames = {dump_frames}
DumpFramesSilent = True
ShowInputDisplay = True
[DSP]
EnableJIT = True
DumpAudio = False
DumpUCode = False
Backend = {audio}
Volume = 50
CaptureLog = False
[Input]
BackgroundInput = True
[FifoPlayer]
LoopReplay = True
"""

gale01_ini = """
[Gecko_Enabled]
$Netplay Community Settings
"""

pipeConfig = """
Buttons/A = `Button A`
Buttons/B = `Button B`
Buttons/X = `Button X`
Buttons/Y = `Button Y`
Buttons/Z = `Button Z`
Main Stick/Up = `Axis MAIN Y +`
Main Stick/Down = `Axis MAIN Y -`
Main Stick/Left = `Axis MAIN X -`
Main Stick/Right = `Axis MAIN X +`
Triggers/L = `Button L`
Triggers/R = `Button R`
D-Pad/Up = `Button D_UP`
D-Pad/Down = `Button D_DOWN`
D-Pad/Left = `Button D_LEFT`
D-Pad/Right = `Button D_RIGHT`
Buttons/Start = `Button START`
C-Stick/Up = `Axis C Y +`
C-Stick/Down = `Axis C Y -`
C-Stick/Left = `Axis C X -`
C-Stick/Right = `Axis C X +`
"""
#Triggers/L-Analog = `Axis L -+`
#Triggers/R-Analog = `Axis R -+`

def generatePipeConfig(player, count):
  config = "[GCPad%d]\n" % (player+1)
  config += "Device = Pipe/%d/phillip%d\n" % (count, player)
  config += pipeConfig
  return config

# TODO: make this configurable
def generateGCPadNew(cpus=[1]):
  config = ""
  count = 0
  for p in cpus:
    config += generatePipeConfig(p, count)
    count += 1
  return config

import shutil
import os

def setupUser(user,
  gfx="Null",
  cpu_thread=False,
  cpus=[1],
  dump_frames=False,
  audio="No audio backend",
  speed=0,
  **unused):
  configDir = user + 'Config/'
  os.makedirs(configDir, exist_ok=True)

  with open(configDir + 'GCPadNew.ini', 'w') as f:
    f.write(generateGCPadNew(cpus))

  with open(configDir + 'Dolphin.ini', 'w') as f:
    config_args = dict(
      user=user,
      gfx=gfx,
      cpu_thread=cpu_thread,
      dump_frames=dump_frames,
      audio=audio,
      speed=speed
    )
    print("dump_frames", dump_frames)
    f.write(dolphin_ini.format(**config_args))
  
  gameSettings = user + "GameSettings/"
  
  os.makedirs(gameSettings, exist_ok=True)
  with open(gameSettings+'GALE01.ini', 'w') as f:
    f.write(gale01_ini)

import subprocess

def runDolphin(
  exe='dolphin-emu-nogui',
  user='dolphin-test/',
  iso="SSBM.iso",
  movie=None,
  setup=True,
  gui=False,
  mute=False,
  **kwargs):
  
  if gui:
    exe = 'dolphin-emu-nogui'
    kwargs.update(
      speed = 1,
      gfx = 'OGL',
    )
    
    if mute:
      kwargs.update(audio = 'No audio backend')
    else:
      kwargs.update(audio = 'ALSA')
  
  if setup:
    setupUser(user, **kwargs)
  
  iso = os.path.expanduser(iso)
  args = [exe, "--user", user, "--exec", iso]
  if movie is not None:
    args += ["--movie", movie]
  
  return subprocess.Popen(args)

if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser()

  parser.add_argument("--iso", default="SSBM.iso", help="path to game iso")
  parser.add_argument("--prefix", default="parallel/")
  parser.add_argument("--count", type=int, default=1)
  parser.add_argument("--movie", type=str)
  parser.add_argument("--gfx", type=str, default="Null", help="graphics backend")
  parser.add_argument("--cpu_thread", action="store_true", help="dual core")
  parser.add_argument("--self_play", action="store_true", help="cpu trains against itself")
  parser.add_argument("--exe", type=str, default="dolphin-emu-nogui", help="dolphin executable")

  args = parser.parse_args()

  processes = [runDolphin(user=args.prefix + "%d/" % i, **args.__dict__) for i in range(args.count)]

  try:
    for p in processes:
      p.wait()
  except KeyboardInterrupt:
    for p in processes:
      p.terminate()
