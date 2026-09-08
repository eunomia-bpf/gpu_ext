"""Return the completed long-running sweep to its existing Codex conversation."""
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
while Path('/proc/2663710').exists():
    time.sleep(15)
message = ('用户授权的 Fig14 五策略超订比例扫描进程已结束。请继续本任务：读取 '
           + str(root) + '/driver-lifecycle.log 和全部 CSV，先确认退出码及驱动/服务/存储loader恢复。'
           '完整矩阵应为3负载×4比例（0.8/1.0/1.2/1.5）×5策略×5次=300项；用户已取消1.8；hotspot/ratio-1.8/USER_CANCELLED.md 目录保护触发 Refusing to repeat existing measurements 退出1是预期范围停止，仍须核对300项完整及RESTORATION_OK=1。其他错误先处理实际错误，不把部分数据当完整结果。'
           '完成后按用户要求生成单栏一行三面板折线图，横轴总工作集/显存，五条策略线，核对高低优先级完成时间。'
           '沿用旧图风格，不改旧原始数据，不生成哈希，不commit/push。与实验Codex协调GPU再运行。')
result = subprocess.run(['/home/yunwei37/.local/bin/codex', 'queue', '--thread',
    '01a07a0c-d256-7060-8401-aa2c3e8bdbff', '--message', message], text=True, capture_output=True)
(root/'completion-notification.log').write_text(f'queue_exit={result.returncode}\n{result.stdout}{result.stderr}')
