pyinstaller --noconfirm  --clean \
--exclude-module PyQt4 \
--exclude-module PyQt4.QtCore \
--exclude-module PyQt4.QtGui \
--hidden-import=h5py.defs \
--hidden-import=h5py.utils \
--hidden-import=h5py.h5ac \
--hidden-import='h5py._proxy' \
--onedir \
test_pyinstaller.py

cp test.avi ./dist/test_pyinstaller