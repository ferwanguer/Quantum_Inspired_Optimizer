import sys
import time
def progress(count, total, status='Processing'):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
    sys.stdout.flush()

for i in range(100):
    progress(i,100)
    time.sleep(0.2)

#De cojones