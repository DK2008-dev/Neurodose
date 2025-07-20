#!/usr/bin/env python3
"""
Complete PDF Generator for Comprehensive Merged Research Paper
Extracts and includes ALL content from the comprehensive markdown file with inline images
"""

import os
import re
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import black, blue, red, green, gray, lightgrey
from reportlab.lib import colors
from PIL import Image as PILImage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompletePDFGenerator:
    def __init__(self, output_file="COMPREHENSIVE_MERGED_RESEARCH_PAPER_COMPLETE.pdf"):
        self.output_file = output_file
        self.doc = SimpleDocTemplate(
            output_file,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
            title="The Complexity Paradox and Augmentation Illusion in EEG-Based Pain Detection"
        )
        
        # Enhanced styles
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
        # Story container
        self.story = []
        
        # Counters
        self.figure_counter = 0
        self.table_counter = 0

    def setup_custom_styles(self):
        """Set up custom paragraph styles for academic formatting"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Author style
        self.styles.add(ParagraphStyle(
            name='AuthorInfo',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            alignment=TA_CENTER,
            textColor=black,
            fontName='Helvetica'
        ))
        
        # Section heading styles
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12,
            alignment=TA_LEFT,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=10,
            spaceBefore=10,
            alignment=TA_LEFT,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubSubHeading',
            parent=self.styles['Heading3'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=8,
            alignment=TA_LEFT,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Abstract style
        self.styles.add(ParagraphStyle(
            name='AbstractStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=12,
            spaceBefore=12,
            alignment=TA_JUSTIFY,
            textColor=black,
            fontName='Helvetica',
            leftIndent=20,
            rightIndent=20
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='MainBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            spaceBefore=6,
            alignment=TA_JUSTIFY,
            textColor=black,
            fontName='Helvetica'
        ))
        
        # List style
        self.styles.add(ParagraphStyle(
            name='BulletList',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            spaceBefore=3,
            alignment=TA_LEFT,
            textColor=black,
            fontName='Helvetica',
            leftIndent=20,
            bulletIndent=10
        ))
        
        # Table caption style
        self.styles.add(ParagraphStyle(
            name='TableTitle',
            parent=self.styles['Normal'],
            fontSize=9,
            spaceAfter=8,
            spaceBefore=8,
            alignment=TA_CENTER,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # Figure caption style
        self.styles.add(ParagraphStyle(
            name='FigureCaption',
            parent=self.styles['Normal'],
            fontSize=9,
            spaceAfter=8,
            spaceBefore=8,
            alignment=TA_CENTER,
            textColor=black,
            fontName='Helvetica-Bold'
        ))
        
        # GitHub section style
        self.styles.add(ParagraphStyle(
            name='GitHubSection',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            spaceBefore=6,
            alignment=TA_LEFT,
            textColor=blue,
            fontName='Helvetica'
        ))

    def clean_text(self, text):
        """Clean and format text for PDF rendering"""
        if not text:
            return ""
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italic
        text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)  # Code
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def parse_table(self, table_lines):
        """Parse markdown table into reportlab table"""
        if not table_lines:
            return None
        
        data = []
        for line in table_lines:
            if '|' in line and not line.strip().startswith('|---'):
                # Split by | and clean up
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if cells:  # Only add non-empty rows
                    data.append(cells)
        
        if not data:
            return None
        
        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        return table

    def add_figure(self, figure_path, caption=None, max_width=15*cm, max_height=10*cm):
        """Add a figure to the PDF with optional caption"""
        try:
            if os.path.exists(figure_path):
                # Get image dimensions
                with PILImage.open(figure_path) as pil_img:
                    img_width, img_height = pil_img.size
                
                # Calculate scaling to fit within max dimensions while maintaining aspect ratio
                width_scale = max_width / img_width
                height_scale = max_height / img_height
                scale = min(width_scale, height_scale, 1.0)  # Don't scale up
                
                final_width = img_width * scale
                final_height = img_height * scale
                
                # Add the image
                img = Image(figure_path, width=final_width, height=final_height)
                self.story.append(img)
                
                # Add caption if provided
                if caption:
                    self.figure_counter += 1
                    caption_text = f"Figure {self.figure_counter}: {caption}"
                    self.story.append(Paragraph(caption_text, self.styles['FigureCaption']))
                
                self.story.append(Spacer(1, 12))
                return True
            else:
                logger.warning(f"Figure not found: {figure_path}")
                return False
        except Exception as e:
            logger.error(f"Error adding figure {figure_path}: {str(e)}")
            return False

    def get_figure_path(self, figure_name):
        """Get the full path to a figure"""
        figures_dir = os.path.join(os.path.dirname(__file__), "figures")
        return os.path.join(figures_dir, figure_name)

    def process_markdown_content(self, content):
        """Process the complete markdown content and convert to PDF elements"""
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Main title
            if line.startswith('# ') and 'Complexity Paradox' in line:
                title_text = self.clean_text(line[2:])
                self.story.append(Paragraph(title_text, self.styles['MainTitle']))
                self.story.append(Spacer(1, 12))
                
            # Authors and affiliations
            elif line.startswith('**Authors:**'):
                authors = self.clean_text(line[12:])
                self.story.append(Paragraph(f"Authors: {authors}", self.styles['AuthorInfo']))
                
            elif line.startswith('**Affiliations:**'):
                affiliations = self.clean_text(line[17:])
                self.story.append(Paragraph(f"Affiliations: {affiliations}", self.styles['AuthorInfo']))
                
            elif line.startswith('**Target Journal:**'):
                journal = self.clean_text(line[19:])
                self.story.append(Paragraph(f"Target Journal: {journal}", self.styles['AuthorInfo']))
                
            elif line.startswith('**Word Count:**'):
                word_count = self.clean_text(line[15:])
                self.story.append(Paragraph(f"Word Count: {word_count}", self.styles['AuthorInfo']))
                self.story.append(Spacer(1, 20))
                
            # Abstract section - use new abstract
            elif line.startswith('## Abstract'):
                self.story.append(Paragraph("Abstract", self.styles['SectionHeading']))
                
                # Add the new abstract content
                new_abstract = """Electroencephalography (EEG) offers a non‑invasive window into nociceptive processing, yet its real‑world value for objective pain assessment remains uncertain. Studies that report 87–91 percent accuracy typically mix data from the same participant across folds, inflating performance.

<b>Objective</b> — Quantify performance under participant‑independent validation, determine why ternary classification collapses, compare classic feature engineering with deep learning, and measure the inflation caused by popular data‑augmentation routines.

<b>Methods</b> — We re‑analysed the open Brain Mediators for Pain dataset (49 participants after quality control). Six pipelines were built: (i) a 78‑feature Random Forest grounded in neurophysiology; (ii) a 645‑feature Random Forest adding wavelet and connectivity descriptors; (iii) three convolutional networks (SimpleEEGNet, EEGNet, ShallowConvNet) on raw signals; (iv) XGBoost with Bayesian hyper‑parameter search; and (v) systematic augmentation with SMOTE, Gaussian noise, frequency warping and temporal shifting. Binary and ternary schemes were tested with strict Leave‑One‑Participant‑Out cross‑validation (LOPOCV).

<b>Results</b> — The 78‑feature Random Forest delivered the best binary accuracy (51.7 ± 4.4 %), edging the 645‑feature model (51.1 ± 6.1 %) and outperforming all CNNs (46.8–48.7 ± 2.7 %). Ternary classification fell to 35.2 ± 5.3 %, close to the 33.3 percent baseline. Augmentation seemed to boost accuracy by 18.3 percent under leaky k‑fold validation but only 2.1 percent under LOPOCV; 79–97 percent of the apparent gains were artefactual. These distortions explain the 35–39 percent gap between literature claims and deployment‑realistic performance.

<b>Conclusions</b> — Under clinical conditions, EEG pain classifiers offer only modest advantages over chance, and greater algorithmic complexity consistently harms generalisation—a complexity paradox. Reported augmentation benefits stem largely from an augmentation illusion. Progress will require participant‑independent benchmarks, subject‑specific calibration, and multimodal fusion rather than deeper networks.

<b>Keywords:</b> EEG, objective pain assessment, machine learning, participant‑independent validation, complexity paradox, augmentation illusion"""
                
                self.story.append(Paragraph(new_abstract, self.styles['AbstractStyle']))
                self.story.append(Spacer(1, 20))
                
                # Skip processing the old abstract content
                j = i + 1
                while j < len(lines) and not lines[j].startswith('## '):
                    j += 1
                i = j - 1
                
            # Section headers with figure insertion
            elif line.startswith('## '):
                header_text = self.clean_text(line[3:])
                self.story.append(Paragraph(header_text, self.styles['SectionHeading']))
                
                # Add relevant figures based on section
                if "Results" in header_text:
                    self.add_figure(self.get_figure_path("complexity_paradox_analysis.png"), 
                                  "Overview of the complexity paradox across multiple dimensions")
                elif "Discussion" in header_text:
                    self.add_figure(self.get_figure_path("literature_gap_comprehensive.png"), 
                                  "Literature performance gap analysis showing methodological factors")
                
            elif line.startswith('### '):
                header_text = self.clean_text(line[4:])
                self.story.append(Paragraph(header_text, self.styles['SubHeading']))
                
                # Add specific figures for key subsections
                if "Complexity Paradox" in header_text:
                    self.add_figure(self.get_figure_path("enhanced_complexity_paradox.png"), 
                                  "Multi-dimensional complexity paradox demonstration")
                elif "Ternary Classification" in header_text:
                    self.add_figure(self.get_figure_path("ternary_failure_comprehensive.png"), 
                                  "Systematic failure of ternary classification across all methods")
                elif "Augmentation Illusion" in header_text:
                    self.add_figure(self.get_figure_path("augmentation_illusion_comprehensive.png"), 
                                  "Quantified augmentation illusion effects under different validation schemes")
                elif "Individual Participant" in header_text:
                    self.add_figure(self.get_figure_path("individual_differences_enhanced.png"), 
                                  "Individual participant performance variability analysis")
                elif "Feature Importance" in header_text:
                    self.add_figure(self.get_figure_path("feature_importance_enhanced.png"), 
                                  "Feature importance analysis showing dominance of simple features")
                
            elif line.startswith('#### '):
                header_text = self.clean_text(line[5:])
                self.story.append(Paragraph(header_text, self.styles['SubSubHeading']))
                
            # Tables
            elif '|' in line and ('---|' in lines[i+1] if i+1 < len(lines) else False):
                # Find the complete table
                table_lines = []
                j = i
                while j < len(lines) and ('|' in lines[j] or not lines[j].strip()):
                    if '|' in lines[j]:
                        table_lines.append(lines[j])
                    j += 1
                
                table = self.parse_table(table_lines)
                if table:
                    self.table_counter += 1
                    # Add table caption if it exists before the table
                    if i > 0 and lines[i-1].strip().startswith('**Table'):
                        caption_text = self.clean_text(lines[i-1].strip())
                        self.story.append(Paragraph(caption_text, self.styles['TableTitle']))
                    
                    self.story.append(table)
                    self.story.append(Spacer(1, 12))
                
                i = j - 1
                
            # List items
            elif line.startswith('- ') or line.startswith('* ') or re.match(r'^\d+\.', line):
                list_text = self.clean_text(line[2:] if line.startswith(('- ', '* ')) else line[line.find('.')+1:].strip())
                bullet = "•" if line.startswith(('- ', '* ')) else f"{line[:line.find('.')+1]}"
                self.story.append(Paragraph(f"{bullet} {list_text}", self.styles['BulletList']))
                
            # Special formatting
            elif line.startswith('**') and line.endswith('**'):
                bold_text = self.clean_text(line)
                self.story.append(Paragraph(bold_text, self.styles['MainBody']))
                
            # Horizontal rules
            elif line.startswith('---'):
                self.story.append(Spacer(1, 12))
                
            # Regular paragraphs
            elif line:
                # Check if this is part of a multi-line paragraph
                paragraph_lines = [line]
                j = i + 1
                while j < len(lines) and lines[j].strip() and not lines[j].startswith(('#', '-', '*', '|', '**Table', '**Figure')):
                    paragraph_lines.append(lines[j].strip())
                    j += 1
                
                paragraph_text = ' '.join(paragraph_lines)
                paragraph_text = self.clean_text(paragraph_text)
                
                if paragraph_text:
                    self.story.append(Paragraph(paragraph_text, self.styles['MainBody']))
                    self.story.append(Spacer(1, 6))
                
                i = j - 1
            
            # Add GitHub section before References
            elif line.startswith('## References'):
                # First add the GitHub Data Availability section
                self.story.append(Paragraph("Data and Code Availability", self.styles['SectionHeading']))
                
                github_content = """<b>Dataset:</b> The OSF "Brain Mediators for Pain" dataset used in this study is publicly available at: https://osf.io/bsv86/

<b>Analysis Code:</b> All analysis code, preprocessing pipelines, and model implementations are available on GitHub:
• Repository: https://github.com/DK2008-dev/Neurodose
• Analysis scripts: /research_paper_analysis/scripts/
• Model implementations: /research_paper_analysis/models/
• Figure generation: /research_paper_analysis/figures/
• Results data: /research_paper_analysis/results/

<b>Reproducibility:</b> Complete instructions for reproducing all results, including environment setup and data processing steps, are provided in the repository README. All hyperparameters, preprocessing parameters, and validation procedures are documented.

<b>Requirements:</b> Python 3.8+, MNE-Python, scikit-learn, TensorFlow, XGBoost. See requirements.txt for complete dependency list."""
                
                self.story.append(Paragraph(github_content, self.styles['GitHubSection']))
                self.story.append(Spacer(1, 20))
                
                # Now add the References section
                header_text = self.clean_text(line[3:])
                self.story.append(Paragraph(header_text, self.styles['SectionHeading']))
            
            i += 1

    def generate_pdf(self, markdown_file):
        """Generate the complete PDF from markdown content"""
        try:
            # Read the complete markdown file
            with open(markdown_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Processing markdown file: {markdown_file}")
            logger.info(f"Content length: {len(content)} characters")
            
            # Process all content
            self.process_markdown_content(content)
            
            logger.info(f"Generated {len(self.story)} story elements")
            logger.info(f"Tables processed: {self.table_counter}")
            
            # Build the PDF
            self.doc.build(self.story)
            
            logger.info(f"PDF generated successfully: {self.output_file}")
            
            # Get file size
            file_size = os.path.getsize(self.output_file) / (1024 * 1024)  # MB
            logger.info(f"PDF size: {file_size:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            return False

def main():
    """Main function to generate the complete comprehensive PDF"""
    
    # Input file path
    markdown_file = "COMPREHENSIVE_MERGED_RESEARCH_PAPER.md"
    
    if not os.path.exists(markdown_file):
        logger.error(f"Markdown file not found: {markdown_file}")
        return False
    
    # Create PDF generator
    generator = CompletePDFGenerator()
    
    # Generate PDF
    success = generator.generate_pdf(markdown_file)
    
    if success:
        print(f"\n✅ SUCCESS: Complete comprehensive PDF generated!")
        print(f"📄 Output file: {generator.output_file}")
        print(f"📊 Tables included: {generator.table_counter}")
        print(f"📝 Story elements: {len(generator.story)}")
        
        # File info
        if os.path.exists(generator.output_file):
            file_size = os.path.getsize(generator.output_file) / (1024 * 1024)
            print(f"📦 File size: {file_size:.2f} MB")
    else:
        print(f"\n❌ FAILED: Could not generate PDF")
        return False
    
    return True

if __name__ == "__main__":
    main()
