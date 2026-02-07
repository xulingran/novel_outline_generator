# Application Icons

This directory should contain the application icons for different platforms.

## Required Icons

### Windows
- **File**: `icon.ico`
- **Format**: ICO file with multiple sizes (16x16, 32x32, 48x48, 256x256)
- **Usage**: Taskbar, window title, executable icon

### macOS
- **File**: `icon.icns`
- **Format**: ICNS file
- **Usage**: Dock, window title, application bundle

### Linux
- **File**: `icon.png`
- **Format**: PNG file, 256x256 or 512x512 pixels
- **Usage**: Window title, application menu

## Creating Icons

### Option 1: Use Online Tools
1. Create a 512x512 PNG image with your design
2. Use online converters:
   - For ICO: https://convertio.co/png-ico/
   - For ICNS: https://cloudconvert.com/png-to-icns

### Option 2: Use ImageMagick
```bash
# Convert PNG to ICO
convert icon.png -define icon:auto-resize=256,48,32,16 icon.ico

# Convert PNG to ICNS (macOS)
mkdir icon.iconset
sips -z 16 16     icon.png --out icon.iconset/icon_16x16.png
sips -z 32 32     icon.png --out icon.iconset/icon_16x16@2x.png
sips -z 32 32     icon.png --out icon.iconset/icon_32x32.png
sips -z 64 64     icon.png --out icon.iconset/icon_32x32@2x.png
sips -z 128 128   icon.png --out icon.iconset/icon_128x128.png
sips -z 256 256   icon.png --out icon.iconset/icon_128x128@2x.png
sips -z 256 256   icon.png --out icon.iconset/icon_256x256.png
sips -z 512 512   icon.png --out icon.iconset/icon_256x256@2x.png
sips -z 512 512   icon.png --out icon.iconset/icon_512x512.png
iconutil -c icns icon.iconset
```

## Icon Design Guidelines

### Theme
- A book or notebook icon
- Color: Blue or green tones
- Style: Modern, flat design

### Size Considerations
- Design at 512x512 pixels
- Avoid thin lines that may disappear at small sizes
- Use high contrast for visibility
- Keep it simple and recognizable

## Temporary Placeholders

If you don't have custom icons yet, you can:
1. Use any square PNG image as `icon.png`
2. Use online converters to generate `icon.ico` and `icon.icns`
3. The application will still work with generic OS default icons
