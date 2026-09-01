from pathlib import Path

path = Path('.github/scripts/temp-fattn-tile-wave64-vkq.py')
text = path.read_text()
text = text.replace('''assert text.count(old) == 2
text = text.replace(old, new)

start = text.index(''', '''assert text.count(old) == 1
text = text.replace(old, new, 1)

sink_old = old.replace('VKQ[jc*', 'VKQ[jc0*')
sink_new = new.replace('VKQ[jc*', 'VKQ[jc0*')
assert text.count(sink_old) == 1
text = text.replace(sink_old, sink_new, 1)

start = text.index(''', 1)
text = text.replace("Path('.github/scripts/temp-fattn-tile-wave64-vkq.py').unlink()", "Path('.github/scripts/temp-fattn-tile-wave64-vkq.py').unlink()\nPath('.github/scripts/temp-fix-fattn-tile-wave64-vkq.py').unlink()")
path.write_text(text)
