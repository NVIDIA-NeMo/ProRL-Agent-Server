"""
Accessibility Tree Simplifier for AI Agents

This module provides utilities to simplify complex accessibility trees (AST) 
into AI-agent-friendly formats. It extracts actionable elements, their coordinates,
and provides context about the desktop environment.

The simplified format helps AI agents:
1. Understand the context of desktop windows and applications
2. Identify actionable items and their types
3. Know how to operate on elements (coordinates, typing locations, etc.)
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import lxml.etree as etree


class AccessibilityTreeSimplifier:
    """Simplifies accessibility trees for AI agent consumption."""
    
    # Define actionable roles that agents can interact with
    ACTIONABLE_ROLES = {
        'push-button', 'toggle-button', 'radio-button', 'check-box',
        'button', 'menu', 'menu-item', 'text', 'entry', 'password-text',
        'combo-box', 'list-item', 'tree-item', 'slider', 'scroll-bar',
        'tab', 'link', 'terminal', 'document-text', 'document-web',
        'tool-bar', 'split-pane', 'page-tab', 'check-menu-item',
        'radio-menu-item', 'spin-button', 'tree-table', 'table-cell',
        'icon', 'canvas', 'drawing-area', 'paragraph', 'label'
    }
    
    # Clickable elements
    CLICKABLE_ROLES = {
        'push-button', 'toggle-button', 'radio-button', 'check-box',
        'button', 'menu', 'menu-item', 'link', 'tab', 'combo-box',
        'list-item', 'tree-item', 'icon', 'tool-bar', 'page-tab',
        'check-menu-item', 'radio-menu-item', 'spin-button', 'scroll-bar',
        'slider', 'tree-table', 'table-cell', 'canvas', 'label'
    }
    
    # Typeable elements
    TYPEABLE_ROLES = {
        'text', 'entry', 'password-text', 'terminal', 'document-text',
        'document-web', 'spin-button', 'combo-box', 'paragraph'
    }
    
    # Namespace mappings for different platforms
    NAMESPACES = {
        'ubuntu': {
            'st': 'https://accessibility.ubuntu.example.org/ns/state',
            'attr': 'https://accessibility.ubuntu.example.org/ns/attributes',
            'cp': 'https://accessibility.ubuntu.example.org/ns/component',
            'act': 'https://accessibility.ubuntu.example.org/ns/action',
            'val': 'https://accessibility.ubuntu.example.org/ns/value',
            'txt': 'https://accessibility.ubuntu.example.org/ns/text',
        },
        'windows': {
            'st': 'https://accessibility.windows.example.org/ns/state',
            'attr': 'https://accessibility.windows.example.org/ns/attributes',
            'cp': 'https://accessibility.windows.example.org/ns/component',
            'class': 'https://accessibility.windows.example.org/ns/class'
        },
        'macos': {
            'st': 'https://accessibility.macos.example.org/ns/state',
            'attr': 'https://accessibility.macos.example.org/ns/attributes',
            'cp': 'https://accessibility.macos.example.org/ns/component',
            'role': 'https://accessibility.macos.example.org/ns/role',
        }
    }
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080, platform: str = 'ubuntu'):
        """
        Initialize the simplifier.
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            platform: Platform type ('ubuntu', 'windows', 'macos')
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.platform = platform
        self.ns = self.NAMESPACES.get(platform, self.NAMESPACES['ubuntu'])
        
    def simplify_tree(self, xml_string: str, max_items: int = 1000) -> Dict[str, Any]:
        """
        Simplify an accessibility tree XML into an AI-agent-friendly format.
        
        Args:
            xml_string: The XML accessibility tree as a string
            max_items: Maximum number of actionable items to extract
            
        Returns:
            Dictionary with simplified tree structure
        """
        try:
            root = etree.fromstring(xml_string.encode('utf-8'))
        except Exception as e:
            return {
                'error': f'Failed to parse XML: {str(e)}',
                'actionable_items': [],
                'screen_organization': {}
            }
        
        # Extract actionable items
        actionable_items = self._extract_actionable_items(root, max_items)
        
        # Extract screen organization
        screen_org = self._extract_screen_organization(root)
        
        # Calculate statistics
        stats = self._calculate_statistics(actionable_items)
        
        return {
            'screen_resolution': {
                'width': self.screen_width,
                'height': self.screen_height
            },
            'screen_organization': screen_org,
            'actionable_items': actionable_items,
            'statistics': stats
        }
    
    def _extract_actionable_items(self, root: etree.Element, max_items: int) -> List[Dict[str, Any]]:
        """Extract actionable items from the accessibility tree."""
        items = []
        seen_items = set()  # Avoid duplicates based on (role, name, coords)
        
        def traverse(element: etree.Element, app_name: str = '', window_name: str = ''):
            """Recursively traverse the tree to find actionable items."""
            if len(items) >= max_items:
                return
            
            tag = element.tag.split('}')[-1]  # Remove namespace
            
            # Update context
            if tag == 'application':
                app_name = element.get('name', '')
            elif tag in ('window', 'frame', 'dialog'):
                window_name = element.get('name', '')
            
            # Check if element is actionable
            if tag in self.ACTIONABLE_ROLES:
                item = self._extract_item_info(element, tag, app_name, window_name)
                if item:
                    # Smart deduplication strategy
                    # Skip standalone labels that are children of list-items (already captured in list-item name)
                    if item['role'] == 'label':
                        # Check if parent is a list-item
                        parent = element.getparent()
                        if parent is not None:
                            parent_tag = parent.tag.split('}')[-1]
                            if parent_tag == 'list-item':
                                # Skip this label, already captured in list-item
                                for child in element:
                                    traverse(child, app_name, window_name)
                                return
                    
                    # Skip standalone icons that are within list-items/buttons
                    if item['role'] == 'icon' and not item['name']:
                        parent = element.getparent()
                        if parent is not None:
                            parent_tag = parent.tag.split('}')[-1]
                            if parent_tag in ['list-item', 'push-button', 'toggle-button']:
                                # Skip unnamed icons within buttons/list-items
                                for child in element:
                                    traverse(child, app_name, window_name)
                                return
                    
                    # Create unique identifier
                    approx_x = item['coords']['pixel_x'] // 10 * 10
                    approx_y = item['coords']['pixel_y'] // 10 * 10
                    item_id = (item['role'], item['name'], approx_x, approx_y)
                    
                    if item_id not in seen_items:
                        items.append(item)
                        seen_items.add(item_id)
            
            # Traverse children
            for child in element:
                traverse(child, app_name, window_name)
        
        traverse(root)
        return items
    
    def _extract_item_info(self, element: etree.Element, role: str, 
                           app_name: str, window_name: str) -> Optional[Dict[str, Any]]:
        """Extract information about a single actionable item."""
        # Get element name - try multiple sources
        name = element.get('name', '') or element.text or ''
        
        # If name is empty or generic, try to find nested label text
        if not name or name.strip() == '':
            # Look for nested label elements
            labels = element.findall('.//label', namespaces=element.nsmap)
            for label in labels:
                label_text = label.get('name', '') or label.text or ''
                if label_text and label_text.strip():
                    name = label_text
                    break
            
            # If still no name, try nested text elements
            if not name or name.strip() == '':
                texts = element.findall('.//text', namespaces=element.nsmap)
                for text_elem in texts:
                    text_content = text_elem.text or ''
                    if text_content and text_content.strip():
                        name = text_content
                        break
        
        name = self._clean_text(name)
        
        # Skip invisible or non-showing elements
        if not self._is_visible(element):
            return None
        
        # Get coordinates
        coords = self._extract_coordinates(element)
        if not coords:
            return None
        
        # Determine action type
        actions = []
        if role in self.CLICKABLE_ROLES:
            actions.append('click')
        if role in self.TYPEABLE_ROLES:
            actions.append('type')
        
        if not actions:
            return None
        
        # Get additional attributes
        value = self._get_value(element)
        description = self._get_description(element)
        states = self._get_states(element)
        
        # Calculate screen position
        position = self._calculate_position(coords['pixel_x'], coords['pixel_y'])
        
        item = {
            'name': name[:100] if name else f"{role}",  # Limit name length
            'role': role,
            'actions': actions,
            'coords': coords,
            'position': position,
            'app': app_name,
            'window': window_name if window_name else None,
        }
        
        # Add optional fields
        if value:
            item['value'] = value
        if description:
            item['description'] = description
        if states:
            item['states'] = states
        
        return item
    
    def _extract_coordinates(self, element: etree.Element) -> Optional[Dict[str, float]]:
        """Extract coordinates from an element."""
        # Try to get screen coordinates
        coord_str = element.get(f'{{{self.ns["cp"]}}}screencoord')
        size_str = element.get(f'{{{self.ns["cp"]}}}size')
        
        if not coord_str or not size_str:
            return None
        
        try:
            # Parse coordinates like "(123, 456)"
            coord_match = re.match(r'\((\d+),\s*(\d+)\)', coord_str)
            size_match = re.match(r'\((\d+),\s*(\d+)\)', size_str)
            
            if not coord_match or not size_match:
                return None
            
            x = int(coord_match.group(1))
            y = int(coord_match.group(2))
            width = int(size_match.group(1))
            height = int(size_match.group(2))
            
            # Calculate center point
            center_x = x + width // 2
            center_y = y + height // 2
            
            # Calculate relative coordinates (0-1 range)
            rel_x = round(center_x / self.screen_width, 4)
            rel_y = round(center_y / self.screen_height, 4)
            
            return {
                'pixel_x': center_x,
                'pixel_y': center_y,
                'x': rel_x,
                'y': rel_y,
                'width': width,
                'height': height,
                'bbox': [x, y, x + width, y + height]  # [left, top, right, bottom]
            }
        except (ValueError, AttributeError):
            return None
    
    def _is_visible(self, element: etree.Element) -> bool:
        """Check if element is visible and showing."""
        showing = element.get(f'{{{self.ns["st"]}}}showing', 'false')
        visible = element.get(f'{{{self.ns["st"]}}}visible', 'false')
        
        # For some platforms, check size
        size_str = element.get(f'{{{self.ns["cp"]}}}size')
        if size_str:
            size_match = re.match(r'\((\d+),\s*(\d+)\)', size_str)
            if size_match:
                width = int(size_match.group(1))
                height = int(size_match.group(2))
                if width <= 0 or height <= 0:
                    return False
        
        return showing == 'true' and visible == 'true'
    
    def _get_value(self, element: etree.Element) -> Optional[str]:
        """Get the value of an element (for inputs, sliders, etc.)."""
        value = element.get(f'{{{self.ns.get("val", "")}}}value')
        return self._clean_text(value) if value else None
    
    def _get_description(self, element: etree.Element) -> Optional[str]:
        """Get the description of an element."""
        desc = element.get(f'{{{self.ns["attr"]}}}description')
        return self._clean_text(desc) if desc else None
    
    def _get_states(self, element: etree.Element) -> List[str]:
        """Get the states of an element (enabled, focused, checked, etc.)."""
        states = []
        state_attrs = ['enabled', 'focused', 'selected', 'checked', 'pressed', 
                      'expanded', 'collapsed', 'editable', 'active']
        
        for state in state_attrs:
            if element.get(f'{{{self.ns["st"]}}}{state}') == 'true':
                states.append(state)
        
        return states
    
    def _calculate_position(self, x: int, y: int) -> str:
        """Calculate the general screen position of an element."""
        # Divide screen into 9 regions
        h_zone = 'left' if x < self.screen_width / 3 else \
                 'right' if x > 2 * self.screen_width / 3 else 'center'
        v_zone = 'top' if y < self.screen_height / 3 else \
                 'bottom' if y > 2 * self.screen_height / 3 else 'middle'
        
        if h_zone == 'center' and v_zone == 'middle':
            return 'center'
        elif h_zone == 'center':
            return f'{v_zone}-center'
        elif v_zone == 'middle':
            return f'{v_zone}-{h_zone}'
        else:
            return f'{v_zone}-{h_zone}'
    
    def _extract_screen_organization(self, root: etree.Element) -> Dict[str, Any]:
        """Extract screen organization information."""
        applications = []
        active_windows = []
        
        for app in root.findall('.//application'):
            app_name = app.get('name', '')
            if not app_name:
                continue
            
            windows = []
            for window in app.findall('.//window') + app.findall('.//frame'):
                window_name = window.get('name', '')
                is_active = window.get(f'{{{self.ns["st"]}}}active') == 'true'
                
                if window_name or is_active:
                    window_info = {
                        'name': window_name,
                        'active': is_active
                    }
                    windows.append(window_info)
                    
                    if is_active:
                        active_windows.append({
                            'app': app_name,
                            'window': window_name
                        })
            
            applications.append({
                'name': app_name,
                'windows': windows
            })
        
        return {
            'applications': applications,
            'active_windows': active_windows
        }
    
    def _calculate_statistics(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about actionable items."""
        stats = {
            'total_items': len(items),
            'clickable': 0,
            'typable': 0,
            'by_role': defaultdict(int),
            'by_app': defaultdict(int),
            'by_position': defaultdict(int)
        }
        
        for item in items:
            if 'click' in item['actions']:
                stats['clickable'] += 1
            if 'type' in item['actions']:
                stats['typable'] += 1
            
            stats['by_role'][item['role']] += 1
            stats['by_app'][item['app']] += 1
            stats['by_position'][item['position']] += 1
        
        # Convert defaultdicts to regular dicts
        stats['by_role'] = dict(stats['by_role'])
        stats['by_app'] = dict(stats['by_app'])
        stats['by_position'] = dict(stats['by_position'])
        
        return stats
    
    def _clean_text(self, text: Optional[str]) -> str:
        """Clean text by removing extra whitespace and control characters."""
        if not text:
            return ''
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def generate_text_summary(self, simplified_tree: Dict[str, Any], max_items: int = 100) -> str:
        """
        Generate a human-readable text summary for AI agents.
        
        Args:
            simplified_tree: Output from simplify_tree()
            max_items: Maximum items to show in summary
            
        Returns:
            Formatted text summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("DESKTOP SCREEN ANALYSIS FOR AI AGENT")
        lines.append("=" * 80)
        lines.append("")
        
        # Screen info
        res = simplified_tree['screen_resolution']
        lines.append(f"Screen Resolution: {res['width']}x{res['height']}")
        lines.append("")
        
        # Screen organization
        lines.append("--- SCREEN ORGANIZATION ---")
        org = simplified_tree['screen_organization']
        lines.append(f"Active Applications: {len(org['applications'])}")
        lines.append("")
        
        # Active windows
        if org['active_windows']:
            lines.append("Active Windows:")
            for win in org['active_windows']:
                lines.append(f"  • {win['app']}: {win['window']}")
            lines.append("")
        
        # Applications with windows
        for app in org['applications'][:20]:  # Show top 20 apps
            if app['windows']:
                lines.append(f"  [{app['name']}]")
                for win in app['windows'][:3]:  # Show up to 3 windows per app
                    status = " (ACTIVE)" if win['active'] else ""
                    if win['name']:
                        lines.append(f"    → {win['name']}{status}")
            else:
                lines.append(f"  [{app['name']}]")
            lines.append("")
        
        # Statistics
        stats = simplified_tree['statistics']
        lines.append("--- ACTIONABLE ITEMS SUMMARY ---")
        lines.append(f"Total: {stats['total_items']}")
        lines.append(f"  • Clickable: {stats['clickable']}")
        lines.append(f"  • Typable: {stats['typable']}")
        lines.append("")
        
        # Top element types
        if stats['by_role']:
            lines.append("Top Element Types:")
            sorted_roles = sorted(stats['by_role'].items(), key=lambda x: x[1], reverse=True)
            for role, count in sorted_roles[:10]:
                lines.append(f"  • {role}: {count}")
            lines.append("")
        
        # Actionable items by app
        lines.append(f"--- ACTIONABLE ITEMS (Top {max_items}) ---")
        lines.append("")
        
        items = simplified_tree['actionable_items'][:max_items]
        
        # Group by app
        items_by_app = defaultdict(list)
        for item in items:
            items_by_app[item['app']].append(item)
        
        for app_name, app_items in sorted(items_by_app.items()):
            lines.append(f"[{app_name}] - {len(app_items)} items")
            for item in app_items[:20]:  # Show top 20 items per app
                icon = "🖱️" if "click" in item['actions'] else ""
                icon = "⌨️" if "type" in item['actions'] else icon
                icon = "🖱️⌨️" if len(item['actions']) > 1 else icon
                
                name = item['name'][:50] if item['name'] else f"[{item['role']}]"
                coords = item['coords']
                pos = item['position']
                
                line = f"  {icon} {name} | {pos} | ({coords['x']:.2f}, {coords['y']:.2f}) | [{coords['pixel_x']}, {coords['pixel_y']}]"
                
                if item.get('value'):
                    line += f" | value: {item['value'][:30]}"
                if item.get('states'):
                    line += f" | {', '.join(item['states'])}"
                
                lines.append(line)
            lines.append("")
        
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def generate_agent_prompt(self, simplified_tree: Dict[str, Any]) -> str:
        """
        Generate a concise prompt for AI agents with essential information.
        
        Args:
            simplified_tree: Output from simplify_tree()
            
        Returns:
            Formatted prompt text
        """
        lines = []
        lines.append("# Current Desktop State")
        lines.append("")
        
        # Screen info
        res = simplified_tree['screen_resolution']
        lines.append(f"**Screen:** {res['width']}x{res['height']}")
        lines.append("")
        
        # Active context
        org = simplified_tree['screen_organization']
        if org['active_windows']:
            lines.append("**Active Window:**")
            for win in org['active_windows']:
                lines.append(f"- {win['app']}: {win['window']}")
            lines.append("")
        
        # Actionable items
        stats = simplified_tree['statistics']
        lines.append(f"**Available Actions:** {stats['total_items']} items ({stats['clickable']} clickable, {stats['typable']} typable)")
        lines.append("")
        
        lines.append("## Actionable Elements:")
        lines.append("")
        
        items = simplified_tree['actionable_items']
        
        for idx, item in enumerate(items[:50], 1):  # Top 50 items
            actions_str = '/'.join(item['actions'])
            name = item['name'][:40] if item['name'] else f"[{item['role']}]"
            coords = item['coords']
            
            line = f"{idx}. **{name}** ({item['role']}) - {actions_str} @ ({coords['pixel_x']}, {coords['pixel_y']})"
            
            if item.get('value'):
                line += f" [value: {item['value'][:20]}]"
            if item.get('states') and any(s in item['states'] for s in ['focused', 'selected', 'checked']):
                line += f" [{', '.join(item['states'])}]"
            
            lines.append(line)
        
        if len(items) > 50:
            lines.append(f"... and {len(items) - 50} more items")
        
        lines.append("")
        lines.append("**Instructions:** Use pixel coordinates (pixel_x, pixel_y) for click/type actions.")
        
        return '\n'.join(lines)


def simplify_accessibility_tree(xml_string: str, 
                                screen_width: int = 1920, 
                                screen_height: int = 1080,
                                platform: str = 'ubuntu',
                                output_format: str = 'json') -> Any:
    """
    Convenience function to simplify an accessibility tree.
    
    Args:
        xml_string: The XML accessibility tree as a string
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        platform: Platform type ('ubuntu', 'windows', 'macos')
        output_format: Output format ('json', 'text', 'prompt')
        
    Returns:
        Simplified tree in the requested format
    """
    simplifier = AccessibilityTreeSimplifier(screen_width, screen_height, platform)
    simplified = simplifier.simplify_tree(xml_string)
    
    if output_format == 'text':
        return simplifier.generate_text_summary(simplified)
    elif output_format == 'prompt':
        return simplifier.generate_agent_prompt(simplified)
    else:
        return simplified