@echo off
setlocal
cd /d %~dp0

REM Clean previous builds to avoid accidentally launching an old EXE
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

py -m pip install -r requirements.txt

REM --add-data ensures the whole local package is bundled even if analysis misses a module
py -m PyInstaller --noconfirm --clean --onedir --windowed --name Miriam ^
  --paths "%cd%" ^
  --add-data "panacea_desktop;panacea_desktop" ^
  --collect-submodules panacea_desktop --collect-submodules PyPDF2 --collect-submodules pypdf --collect-submodules docx ^
  --hidden-import panacea_desktop.core.llm_client ^
  --hidden-import panacea_desktop.core.engine ^
  --hidden-import panacea_desktop.core.db ^
  --hidden-import panacea_desktop.core.rag ^
  --hidden-import panacea_desktop.core.doc_extract ^
  --hidden-import panacea_desktop.core.prompts ^
  --hidden-import panacea_desktop.core.defaults ^
  --hidden-import panacea_desktop.core.utils ^
  --hidden-import panacea_desktop.qt_app ^
  --collect-submodules PySide6 ^
  --collect-data PySide6 ^
  run_app.py

echo.
echo Build finished. Check dist\Miriam\Miriam.exe
pause
