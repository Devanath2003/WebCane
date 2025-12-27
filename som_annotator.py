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
        self.box_thickness = 2
        self.text_color = "#FF000014"
        self.text_bg_color = '#FFFFFF'
        self.font_size = 18
        
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
        if not screenshot_bytes or not elements:
            return screenshot_bytes, []
        
        try:
            image = Image.open(io.BytesIO(screenshot_bytes))
            draw = ImageDraw.Draw(image)
            
            # Draw each element using its INDEX in the list as the label
            for idx, element in enumerate(elements):
                # We pass 'str(idx)' as the label to draw
                self._draw_box_with_id(draw, element, self.box_color, self.font, label=str(idx))
            
            output = io.BytesIO()
            image.save(output, format='PNG')
            return output.getvalue(), elements
            
        except Exception as e:
            print(f"❌ Failed to annotate screenshot: {e}")
            return screenshot_bytes, []
    
    def _draw_box_with_id(
        self, 
        draw: ImageDraw.ImageDraw, 
        element: Dict, 
        color: str, 
        font: ImageFont.ImageFont,
        label: str = None  # <--- New Argument
    ):
        try:
            bbox = element['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # Use provided label (index) or fallback to element ID
            display_text = label if label is not None else str(element['id'])
            
            # Draw rectangle
            draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=self.box_thickness)
            
            # ... (Rest of your drawing code, but use `display_text` variable) ...
            
            # Example for text drawing part:
            draw.text((x + 5, y + 5), display_text, fill=color, font=font)

        except Exception as e:
            print(f"⚠️ Draw error: {e}")
    
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
        Filter elements but DO NOT overwrite their IDs.
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
        
        # Just return the sliced list. Do NOT reassign 'id'.
        return sorted_elements[:max_elements]
    
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