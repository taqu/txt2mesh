import sys
import time
import datetime
import webbrowser

if len(sys.argv) <= 1:
    exit()

url = sys.argv[1]
for i in range(12):
    browse = webbrowser.get()
    browse.open(url)
    print(i, datetime.datetime.today())
    time.sleep(60*60)