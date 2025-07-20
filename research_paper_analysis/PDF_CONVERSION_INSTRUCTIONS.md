
# PDF Conversion Instructions

## Method 1: Browser Print-to-PDF (Recommended)
1. Open `EEG_Pain_Classification_Research_Paper.html` in your browser
2. Press `Ctrl+P` (Windows) or `Cmd+P` (Mac)
3. Select "Save as PDF" or "Microsoft Print to PDF"
4. Adjust settings:
   - Paper size: A4 or Letter
   - Margins: Default or Custom (0.5-1 inch)
   - Headers/Footers: Optional
5. Click "Save" and choose location

## Method 2: Online HTML-to-PDF Converters
- **PDF24**: https://tools.pdf24.org/en/html-to-pdf
- **ILovePDF**: https://www.ilovepdf.com/html-to-pdf
- **SmallPDF**: https://smallpdf.com/html-to-pdf
- **ConvertIO**: https://convertio.co/html-pdf/

## Method 3: Command Line Tools
```bash
# Using wkhtmltopdf (if installed)
wkhtmltopdf EEG_Pain_Classification_Research_Paper.html research_paper.pdf

# Using Pandoc (if installed)
pandoc RESEARCH_PAPER_DRAFT.md -o research_paper.pdf --pdf-engine=xelatex
```

## Professional PDF Features
✅ Academic formatting with Times New Roman font
✅ Proper page breaks and spacing
✅ Professional title page with key findings
✅ Table formatting with borders and headers
✅ Figure reference placeholders
✅ Print-optimized CSS styling
✅ Consistent margins and typography

## Quality Tips
- Use 300 DPI for high-quality printing
- Ensure all content fits within page margins
- Preview before final conversion
- Check table formatting in PDF output
