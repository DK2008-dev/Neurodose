#!/usr/bin/env python3
"""
Generate Complete PDF from Refined Comprehensive Research Paper
Creates publication-ready PDF with refined language, inline figures, and academic formatting
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import black, blue, red, green
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import re
from datetime import datetime
from PIL import Image as PILImage

class RefinedPaperPDFGenerator:
    def __init__(self, markdown_file, output_file):
        self.markdown_file = Path(markdown_file)
        self.output_file = Path(output_file)
        self.story = []
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        self.figures_dir = Path("figures")  # Relative to current directory
        
    def setup_custom_styles(self):
        """Setup custom styles for academic paper formatting"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Author style
        self.styles.add(ParagraphStyle(
            name='Author',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Abstract style
        self.styles.add(ParagraphStyle(
            name='Abstract',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=0.5*inch,
            rightIndent=0.5*inch,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=15,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Table caption
        self.styles.add(ParagraphStyle(
            name='TableCaption',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            spaceBefore=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Figure caption
        self.styles.add(ParagraphStyle(
            name='FigureCaption',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=12,
            spaceBefore=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique'
        ))
        
        # Body text with refined formatting
        self.styles.add(ParagraphStyle(
            name='RefinedBody',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))

    def read_markdown_content(self):
        """Read and return the refined markdown content"""
        try:
            with open(self.markdown_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✓ Successfully read {len(content)} characters from {self.markdown_file}")
            return content
        except FileNotFoundError:
            print(f"✗ Error: Could not find {self.markdown_file}")
            return None
        except Exception as e:
            print(f"✗ Error reading file: {str(e)}")
            return None

    def find_and_add_figure(self, figure_name, caption=""):
        """Find and add a figure to the story"""
        possible_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
        figure_found = False
        
        for ext in possible_extensions:
            figure_path = self.figures_dir / f"{figure_name}{ext}"
            if figure_path.exists():
                try:
                    # Verify image can be opened
                    if ext.lower() in ['.png', '.jpg', '.jpeg']:
                        test_img = PILImage.open(figure_path)
                        test_img.close()
                    
                    # Add image to story
                    img = Image(str(figure_path), width=5*inch, height=3.5*inch)
                    self.story.append(img)
                    
                    # Add caption
                    if caption:
                        caption_para = Paragraph(caption, self.styles['FigureCaption'])
                        self.story.append(caption_para)
                    
                    self.story.append(Spacer(1, 12))
                    figure_found = True
                    print(f"✓ Added figure: {figure_path}")
                    break
                    
                except Exception as e:
                    print(f"✗ Error loading figure {figure_path}: {str(e)}")
                    continue
        
        if not figure_found:
            print(f"⚠ Figure not found: {figure_name}")
            # Add placeholder text
            placeholder = Paragraph(f"[Figure: {figure_name}]", self.styles['FigureCaption'])
            self.story.append(placeholder)

    def create_table_from_markdown(self, table_text):
        """Convert markdown table to ReportLab table"""
        lines = table_text.strip().split('\n')
        if len(lines) < 3:  # Need at least header, separator, and one data row
            return None
            
        # Parse header
        header = [cell.strip() for cell in lines[0].split('|')[1:-1]]
        
        # Parse data rows (skip separator line)
        data_rows = []
        for line in lines[2:]:
            if line.strip():
                row = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(row) == len(header):  # Ensure consistent column count
                    data_rows.append(row)
        
        if not data_rows:
            return None
            
        # Create table data
        table_data = [header] + data_rows
        
        # Create table with styling
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        return table

    def process_content_section(self, content):
        """Process a section of content and add to story"""
        lines = content.split('\n')
        current_table = []
        in_table = False
        processed_lines = 0
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if not in_table:
                    self.story.append(Spacer(1, 6))
                continue
            
            processed_lines += 1
            
            # Check for table
            if '|' in line and (line.startswith('|') or line.count('|') >= 2):
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
                continue
            else:
                # End of table
                if in_table:
                    table_text = '\n'.join(current_table)
                    table = self.create_table_from_markdown(table_text)
                    if table:
                        self.story.append(table)
                        self.story.append(Spacer(1, 12))
                    in_table = False
                    current_table = []
            
            # Process different line types
            if line.startswith('# '):
                # Main title
                title = line[2:].strip()
                title_para = Paragraph(title, self.styles['CustomTitle'])
                self.story.append(title_para)
                self.story.append(Spacer(1, 20))
                
            elif line.startswith('## '):
                # Section heading
                heading = line[3:].strip()
                heading_para = Paragraph(heading, self.styles['SectionHeading'])
                self.story.append(heading_para)
                
            elif line.startswith('### '):
                # Subsection heading
                subheading = line[4:].strip()
                subheading_para = Paragraph(subheading, self.styles['SubsectionHeading'])
                self.story.append(subheading_para)
                
            elif line.startswith('**Table'):
                # Table caption
                caption = line.replace('**', '').strip()
                caption_para = Paragraph(caption, self.styles['TableCaption'])
                self.story.append(caption_para)
                
            elif line.startswith('**') and line.endswith('**'):
                # Bold text/emphasis
                text = line[2:-2].strip()
                if text:  # Make sure there's content
                    bold_para = Paragraph(f"<b>{text}</b>", self.styles['RefinedBody'])
                    self.story.append(bold_para)
                
            elif line.startswith('- ') or line.startswith('* '):
                # Bullet points
                bullet_text = line[2:].strip()
                bullet_para = Paragraph(f"• {bullet_text}", self.styles['RefinedBody'])
                self.story.append(bullet_para)
                
            elif line.startswith(('1. ', '2. ', '3. ', '4. ', '5. ')):
                # Numbered lists
                num_text = line[3:].strip()
                num_para = Paragraph(f"{line[:2]} {num_text}", self.styles['RefinedBody'])
                self.story.append(num_para)
                
            elif line.startswith('*') and line.endswith('*') and not line.startswith('**'):
                # Italic text
                text = line[1:-1].strip()
                if text:  # Make sure there's content
                    italic_para = Paragraph(f"<i>{text}</i>", self.styles['RefinedBody'])
                    self.story.append(italic_para)
                
            else:
                # Regular paragraph
                if line and not line.startswith('---'):  # Skip horizontal rules
                    # Process inline formatting
                    processed_line = self.process_inline_formatting(line)
                    para = Paragraph(processed_line, self.styles['RefinedBody'])
                    self.story.append(para)
        
        # Handle final table if exists
        if in_table and current_table:
            table_text = '\n'.join(current_table)
            table = self.create_table_from_markdown(table_text)
            if table:
                self.story.append(table)
                self.story.append(Spacer(1, 12))
        
        print(f"✓ Processed {processed_lines} lines, created {len(self.story)} story elements so far")

    def process_inline_formatting(self, text):
        """Process inline markdown formatting"""
        # Bold text
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Italic text
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        # Code/monospace
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
        return text

    def add_strategic_figures(self):
        """Add figures at strategic points in the document"""
        print("Adding strategic figures to enhance the refined paper...")
        
        # Add complexity paradox visualization
        self.find_and_add_figure("enhanced_complexity_paradox", 
                                "Figure 1: Multi-dimensional complexity paradox showing performance degradation with increased sophistication")
        
        # Add augmentation illusion figure
        self.find_and_add_figure("augmentation_illusion_comprehensive", 
                                "Figure 2: The augmentation illusion - apparent gains under k-fold validation versus reality under LOPOCV")
        
        # Add individual variability figure
        self.find_and_add_figure("individual_differences_enhanced", 
                                "Figure 3: Massive individual differences in EEG pain classification performance")
        
        # Add feature importance visualization
        self.find_and_add_figure("feature_importance_enhanced", 
                                "Figure 4: Simple spectral features dominate over sophisticated measures")
        
        # Add performance gap analysis
        self.find_and_add_figure("literature_gap_comprehensive", 
                                "Figure 5: The 35-39% performance gap between literature claims and rigorous validation")
        
        # Add ternary classification failure
        self.find_and_add_figure("ternary_failure_comprehensive", 
                                "Figure 6: Systematic failure of ternary pain classification across all methods")

    def generate_pdf(self):
        """Generate the complete PDF with refined content"""
        print("Starting PDF generation with refined language and comprehensive content...")
        
        # Read the refined markdown content
        content = self.read_markdown_content()
        if not content:
            return False
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(self.output_file),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Split content into sections and process with strategic figure placement
        sections = content.split('\n## ')
        
        for i, section in enumerate(sections):
            if i == 0:
                # First section includes title and abstract
                self.process_content_section(section)
            else:
                # Add section marker back
                section = '## ' + section
                self.process_content_section(section)
                
                # Add figures at strategic points
                if 'Results' in section and 'complexity paradox' in section.lower():
                    self.find_and_add_figure("enhanced_complexity_paradox", 
                                            "Figure 1: Multi-dimensional complexity paradox showing performance degradation with increased sophistication")
                elif 'augmentation illusion' in section.lower():
                    self.find_and_add_figure("augmentation_illusion_comprehensive", 
                                            "Figure 2: The augmentation illusion - apparent gains under k-fold validation versus reality under LOPOCV")
                elif 'individual' in section.lower() and 'variability' in section.lower():
                    self.find_and_add_figure("individual_differences_enhanced", 
                                            "Figure 3: Massive individual differences in EEG pain classification performance")
                elif 'ternary' in section.lower() and 'classification' in section.lower():
                    self.find_and_add_figure("ternary_failure_comprehensive", 
                                            "Figure 4: Systematic failure of ternary pain classification across all methods")
                elif 'feature importance' in section.lower():
                    self.find_and_add_figure("feature_importance_enhanced", 
                                            "Figure 5: Simple spectral features dominate over sophisticated measures")
                elif 'literature' in section.lower() and 'gap' in section.lower():
                    self.find_and_add_figure("literature_gap_comprehensive", 
                                            "Figure 6: The 35-39% performance gap between literature claims and rigorous validation")
        
        try:
            # Build the PDF
            doc.build(self.story)
            
            # Get file size
            file_size = self.output_file.stat().st_size / (1024*1024)  # MB
            
            print(f"\n✓ SUCCESS: Refined comprehensive PDF generated!")
            print(f"📄 File: {self.output_file}")
            print(f"📊 Size: {file_size:.2f} MB")
            print(f"📝 Elements: {len(self.story)} story elements")
            print(f"🕒 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error generating PDF: {str(e)}")
            return False

def main():
    """Main function to generate refined comprehensive PDF"""
    
    # File paths
    markdown_file = "COMPREHENSIVE_MERGED_RESEARCH_PAPER_REFINED_20250719_0130.md"
    output_file = "COMPREHENSIVE_MERGED_RESEARCH_PAPER_REFINED_COMPLETE_20250719_0130.pdf"
    
    print("=== Refined Comprehensive Research Paper PDF Generator ===")
    print(f"Source: {markdown_file}")
    print(f"Output: {output_file}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if source file exists
    if not Path(markdown_file).exists():
        print(f"✗ Error: Source file not found: {markdown_file}")
        return
    
    # Generate PDF
    generator = RefinedPaperPDFGenerator(markdown_file, output_file)
    success = generator.generate_pdf()
    
    if success:
        print("\n🎉 Refined comprehensive PDF generation completed successfully!")
        print("📋 The PDF includes:")
        print("   • Refined, conversational language throughout")
        print("   • All original sections and content intact") 
        print("   • Comprehensive tables and analysis")
        print("   • Strategic figure placement")
        print("   • Updated single author information")
        print("   • Public repository section")
        print("   • Professional academic formatting")
    else:
        print("\n❌ PDF generation failed. Check error messages above.")

if __name__ == "__main__":
    main()
