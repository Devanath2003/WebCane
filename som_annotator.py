"""
SoM (Set-of-Mark) Annotator
Draws numbered boxes on interactive elements for vision LLM analysis
"""

from PIL import Image, ImageDraw, ImageFont
import io
from typing import List, Dict, Tuple, Optional


class SoMAnnotator:
    """Annotate screenshots with numbered boxes for vision-based automation"""
    
    def __init__(self):
        """Initialize drawing parameters and settings"""
        # Visual style parameters
        self.box_color = '#FF0000'  # Bright red
        self.box_thickness = 4
        self.text_color = '#FF0000'
        self.text_bg_color = '#FFFFFF'
        self.font_size = 28
        
        # Try to load a good font, fallback to default
        self.font = self._load_font()
        
        print("✅ SoMAnnotator initialized")
    
    def _load_font(self):
        """Load font for text annotations"""
        try:
            # Try common font paths
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "C:\\Windows\\Fonts\\arial.ttf",  # Windows
                "arial.ttf",
                "DejaVuSans.ttf"
            ]
            
            for font_path in font_paths:
                try:
                    return ImageFont.truetype(font_path, self.font_size)
                except:
                    continue
            
            # Fallback to default font
            print("⚠️  Using default font (no TrueType font found)")
            return ImageFont.load_default()
            
        except Exception as e:
            print(f"⚠️  Font loading error: {e}")
            return ImageFont.load_default()
    
    def create_annotated_screenshot(
        self, 
        screenshot_bytes: bytes, 
        elements: List[Dict]
    ) -> Tuple[bytes, List[Dict]]:
        """
        Create annotated screenshot with numbered boxes on all elements
        
        Args:
            screenshot_bytes: Raw screenshot from Playwright
            elements: List of DOM elements from DOMExtractor
            
        Returns:
            Tuple of (annotated_image_bytes, element_mapping)
        """
        if not screenshot_bytes:
            print("❌ No screenshot provided")
            return b'', []
        
        if not elements:
            print("❌ No elements to annotate")
            return screenshot_bytes, []
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            # Create drawing context
            draw = ImageDraw.Draw(image)
            
            # Draw each element
            annotated_count = 0
            for element in elements:
                self._draw_box_with_id(draw, element, self.box_color, self.font)
                annotated_count += 1
            
            # Convert back to bytes
            output = io.BytesIO()
            image.save(output, format='PNG')
            annotated_bytes = output.getvalue()
            
            print(f"✅ Annotated {annotated_count} elements")
            
            return annotated_bytes, elements
            
        except Exception as e:
            print(f"❌ Failed to annotate screenshot: {e}")
            return screenshot_bytes, []
    
    def _draw_box_with_id(
        self, 
        draw: ImageDraw.ImageDraw, 
        element: Dict, 
        color: str, 
        font: ImageFont.ImageFont
    ):
        """
        Draw a numbered box around an element
        
        Args:
            draw: PIL ImageDraw object
            element: Element dict with bbox and id
            color: Box color (hex string)
            font: Font for text
        """
        try:
            bbox = element['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            element_id = element['id']
            
            # Draw rectangle
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline=color,
                width=self.box_thickness
            )
            
            # Prepare ID text
            id_text = str(element_id)
            
            # Get text size for background
            try:
                # For newer Pillow versions
                text_bbox = draw.textbbox((0, 0), id_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                # Fallback for older Pillow versions
                text_width, text_height = draw.textsize(id_text, font=font)
            
            # Text position (top-left corner with padding)
            text_x = x + 5
            text_y = y + 5
            
            # Draw semi-transparent white background for text
            bg_padding = 4
            draw.rectangle(
                [
                    (text_x - bg_padding, text_y - bg_padding),
                    (text_x + text_width + bg_padding, text_y + text_height + bg_padding)
                ],
                fill=self.text_bg_color,
                outline=color,
                width=2
            )
            
            # Draw ID number
            draw.text(
                (text_x, text_y),
                id_text,
                fill=color,
                font=font
            )
            
        except Exception as e:
            print(f"⚠️  Failed to draw element {element.get('id', '?')}: {e}")
    
    def filter_and_annotate(
        self,
        screenshot_bytes: bytes,
        elements: List[Dict],
        max_elements: int = 20
    ) -> Tuple[bytes, List[Dict]]:
        """
        Create annotated screenshot with filtered elements for clarity
        
        Args:
            screenshot_bytes: Raw screenshot
            elements: Full list of elements
            max_elements: Maximum number of elements to annotate
            
        Returns:
            Tuple of (annotated_image_bytes, filtered_elements)
        """
        if not elements:
            return screenshot_bytes, []
        
        try:
            # Filter elements
            filtered_elements = self._filter_elements(elements, max_elements)
            
            print(f"📊 Filtered {len(elements)} → {len(filtered_elements)} elements")
            
            # Annotate with filtered list
            return self.create_annotated_screenshot(screenshot_bytes, filtered_elements)
            
        except Exception as e:
            print(f"❌ Filter and annotate failed: {e}")
            return screenshot_bytes, elements
    
    def _filter_elements(
        self,
        elements: List[Dict],
        max_elements: int
    ) -> List[Dict]:
        """
        Filter elements for cleaner annotation
        
        Priority:
        1. Remove very small elements (< 10x10)
        2. Prioritize elements near top of page
        3. Limit to max_elements
        
        Args:
            elements: Full element list
            max_elements: Maximum to return
            
        Returns:
            Filtered element list
        """
        # Filter out very small elements
        valid_elements = [
            el for el in elements
            if el['bbox']['w'] >= 10 and el['bbox']['h'] >= 10
        ]
        
        # Sort by vertical position (top to bottom) then horizontal (left to right)
        sorted_elements = sorted(
            valid_elements,
            key=lambda el: (el['bbox']['y'], el['bbox']['x'])
        )
        
        # Take first max_elements
        filtered = sorted_elements[:max_elements]
        
        # Reassign sequential IDs for clarity
        for idx, element in enumerate(filtered):
            element['original_id'] = element['id']
            element['id'] = idx
        
        return filtered
    
    def create_multi_color_annotation(
        self,
        screenshot_bytes: bytes,
        elements: List[Dict],
        color_scheme: str = 'rainbow'
    ) -> Tuple[bytes, List[Dict]]:
        """
        Create annotation with multiple colors for better distinction
        
        Args:
            screenshot_bytes: Raw screenshot
            elements: Element list
            color_scheme: 'rainbow' or 'alternate'
            
        Returns:
            Tuple of (annotated_image_bytes, elements)
        """
        if not screenshot_bytes or not elements:
            return screenshot_bytes, elements
        
        try:
            image = Image.open(io.BytesIO(screenshot_bytes))
            draw = ImageDraw.Draw(image)
            
            # Color schemes
            if color_scheme == 'rainbow':
                colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF']
            else:  # alternate
                colors = ['#FF0000', '#00FF00']
            
            # Draw each element with cycling colors
            for idx, element in enumerate(elements):
                color = colors[idx % len(colors)]
                self._draw_box_with_id(draw, element, color, self.font)
            
            # Convert to bytes
            output = io.BytesIO()
            image.save(output, format='PNG')
            
            print(f"✅ Created multi-color annotation with {len(elements)} elements")
            
            return output.getvalue(), elements
            
        except Exception as e:
            print(f"❌ Multi-color annotation failed: {e}")
            return screenshot_bytes, elements


# Interactive test
if __name__ == "__main__":
    from dom_extractor import DOMExtractor
    
    print("=" * 70)
    print("SoM ANNOTATOR TEST")
    print("Set-of-Mark: Draw numbered boxes on interactive elements")
    print("=" * 70)
    
    extractor = DOMExtractor()
    annotator = SoMAnnotator()
    
    try:
        # User input
        url = input("\nEnter URL to annotate (e.g., github.com): ").strip()
        
        if not url:
            print("❌ No URL provided")
            exit(1)
        
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        print("\n" + "=" * 70)
        
        # Start browser
        print(f"\n🌐 Starting browser and navigating to {url}...")
        if not extractor.start_browser(headless=False):
            exit(1)
        
        if not extractor.navigate(url):
            exit(1)
        
        # Extract elements
        print("\n🔍 Extracting DOM elements...")
        elements = extractor.extract_elements()
        
        if not elements:
            print("❌ No elements found")
            exit(1)
        
        print(f"✅ Found {len(elements)} interactive elements")
        
        # Take screenshot
        print("\n📸 Taking screenshot...")
        screenshot = extractor.take_screenshot()
        
        if not screenshot:
            print("❌ Screenshot failed")
            exit(1)
        
        print("✅ Screenshot captured")
        
        # Test 1: Full annotation
        print("\n" + "=" * 70)
        print("TEST 1: Full Annotation (all elements)")
        print("=" * 70)
        
        annotated_img, mapping = annotator.create_annotated_screenshot(
            screenshot,
            elements
        )
        
        with open('som_annotated.png', 'wb') as f:
            f.write(annotated_img)
        
        print(f"\n✅ Saved to: som_annotated.png")
        print(f"   📊 Annotated {len(mapping)} elements")
        print(f"   🔍 Open this file to verify box positions")
        
        # Test 2: Filtered annotation
        print("\n" + "=" * 70)
        print("TEST 2: Filtered Annotation (first 10 elements)")
        print("=" * 70)
        
        filtered_img, filtered_elements = annotator.filter_and_annotate(
            screenshot,
            elements,
            max_elements=10
        )
        
        with open('som_filtered.png', 'wb') as f:
            f.write(filtered_img)
        
        print(f"\n✅ Saved to: som_filtered.png")
        print(f"   📊 Annotated {len(filtered_elements)} elements (filtered)")
        
        # Show filtered element details
        print("\n📋 Filtered Elements:")
        for el in filtered_elements[:5]:
            text = el['text'][:30] if el['text'] else '(no text)'
            print(f"   [{el['id']}] {el['tag']:8s} '{text}' at ({el['bbox']['x']}, {el['bbox']['y']})")
        
        if len(filtered_elements) > 5:
            print(f"   ... and {len(filtered_elements) - 5} more")
        
        # Test 3: Multi-color annotation
        print("\n" + "=" * 70)
        print("TEST 3: Multi-Color Annotation (first 10 with rainbow colors)")
        print("=" * 70)
        
        rainbow_img, rainbow_elements = annotator.create_multi_color_annotation(
            screenshot,
            elements[:10],
            color_scheme='rainbow'
        )
        
        with open('som_rainbow.png', 'wb') as f:
            f.write(rainbow_img)
        
        print(f"\n✅ Saved to: som_rainbow.png")
        print(f"   🌈 Multi-color annotation with {len(rainbow_elements)} elements")
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"\n✅ Generated 3 annotated screenshots:")
        print(f"   1. som_annotated.png  - All {len(elements)} elements")
        print(f"   2. som_filtered.png   - Top 10 elements (filtered)")
        print(f"   3. som_rainbow.png    - Top 10 with colors")
        print(f"\n💡 Open these files to verify:")
        print(f"   - Boxes are correctly positioned")
        print(f"   - Numbers are clearly visible")
        print(f"   - No overlapping issues")
        
        # Wait for user
        print("\n" + "=" * 70)
        input("⏸️  Press Enter to close browser...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        extractor.close()
        print("\n✅ Test complete!")