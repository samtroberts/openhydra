OpenHydra Desktop — Icon Generation
=====================================

This directory should contain the application icons required by Tauri for
packaging on macOS, Windows, and Linux.

Prerequisites
-------------
The Tauri CLI can generate all required icon sizes from a single source image.

Steps
-----

1. Place a high-quality 1024x1024 PNG here and name it icon.png.
   The PNG should have a transparent background and square dimensions.

2. From the desktop/ directory, run:

       npx tauri icon src-tauri/icons/icon.png

   This generates the following files automatically:

       icons/32x32.png
       icons/128x128.png
       icons/128x128@2x.png
       icons/icon.icns       (macOS)
       icons/icon.ico        (Windows)
       icons/Square30x30Logo.png
       icons/Square44x44Logo.png
       icons/Square71x71Logo.png
       icons/Square89x89Logo.png
       icons/Square107x107Logo.png
       icons/Square142x142Logo.png
       icons/Square150x150Logo.png
       icons/Square284x284Logo.png
       icons/Square310x310Logo.png
       icons/StoreLogo.png

3. Commit the generated icons before running npm run build.

Quick start (macOS/Linux)
--------------------------
If you have ImageMagick available you can generate a placeholder icon:

    convert -size 1024x1024 xc:#0a0a0a \
      -fill '#00d4b8' -font Helvetica-Bold -pointsize 600 \
      -gravity Center -annotate 0 'H' \
      src-tauri/icons/icon.png

Then run step 2 above.

Notes
------
- tauri.conf.json references: icons/32x32.png, icons/128x128.png,
  icons/128x128@2x.png, icons/icon.icns, icons/icon.ico
- The trayIcon entry references icons/icon.png (the 1024x1024 source)
- Tauri will error during build if any referenced icon file is missing
