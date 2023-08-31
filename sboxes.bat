del /Q *.lib *.exp *.obj sboxes*.exe sboxes*32 *sboxes*64 sboxes*.log
cl sboxes_deseval.c sboxes_main.c /DWIN32 /Ot /Ob2 /Wp64 /arch:SSE /arch:SSE2 /arch:AVX "/wd 4068" 
if not errorlevel 1 sboxes_deseval.exe %*
