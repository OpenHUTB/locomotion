from dm_control.viewer import application
from myo_dm_adapter import MyoDmAdapter

env = MyoDmAdapter("myoHandReorient8P-v0")   # 任意 MyoSuite 任务
app = application.Application(title="MyoSuite in Puppeteer")
