import ctypes
import os
import platform
import shlex
import json
import subprocess, signal
import time
from pathlib import Path
from typing import Any, Optional, Sequence
from typing import List, Dict, Tuple, Literal
import concurrent.futures

# Server version - increment this to verify server reload
SERVER_VERSION = "2025.11.29.28"

# Debug flag for panel text extraction (set to False for production)
DEBUG_PANEL_TEXT = False
DEBUG_LOG_FILE = "/tmp/panel_debug.log"

def debug_log(msg):
    if DEBUG_PANEL_TEXT:
        with open(DEBUG_LOG_FILE, "a") as f:
            f.write(msg + "\n")

import Xlib
import lxml.etree
import pyautogui
import requests
import re
from PIL import Image, ImageGrab
from Xlib import display, X
from flask import Flask, request, jsonify, send_file, abort  # , send_from_directory
from lxml.etree import _Element

platform_name: str = platform.system()

if platform_name == "Linux":
    import pyatspi
    from pyatspi import Accessible, StateType, STATE_SHOWING
    from pyatspi import Action as ATAction
    from pyatspi import Component  # , Document
    from pyatspi import Text as ATText
    from pyatspi import Value as ATValue

    BaseWrapper = Any

elif platform_name == "Windows":
    from pywinauto import Desktop
    from pywinauto.base_wrapper import BaseWrapper
    import pywinauto.application
    import win32ui, win32gui

    Accessible = Any

elif platform_name == "Darwin":
    import plistlib

    import AppKit
    import ApplicationServices
    import Foundation
    import Quartz
    import oa_atomacos

    Accessible = Any
    BaseWrapper = Any

else:
    # Platform not supported
    Accessible = None
    BaseWrapper = Any

from pyxcursor import Xcursor

# todo: need to reformat and organize this whole file

app = Flask(__name__)

pyautogui.PAUSE = 0
pyautogui.DARWIN_CATCH_UP_TIME = 0

TIMEOUT = 1800  # seconds

logger = app.logger
recording_process = None  # fixme: this is a temporary solution for recording, need to be changed to support multiple-process
recording_path = "/tmp/recording.mp4"


@app.route('/setup/execute', methods=['POST'])
@app.route('/execute', methods=['POST'])
def execute_command():
    data = request.json
    # The 'command' key in the JSON request should contain the command to be executed.
    shell = data.get('shell', False)
    command = data.get('command', "" if shell else [])

    if isinstance(command, str) and not shell:
        command = shlex.split(command)

    # Expand user directory
    for i, arg in enumerate(command):
        if arg.startswith("~/"):
            command[i] = os.path.expanduser(arg)

    # Execute the command without any safety checks.
    try:
        if platform_name == "Windows":
            flags = subprocess.CREATE_NO_WINDOW
        else:
            flags = 0
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            text=True,
            timeout=120,
            creationflags=flags,
        )
        return jsonify({
            'status': 'success',
            'output': result.stdout,
            'error': result.stderr,
            'returncode': result.returncode
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/setup/execute_with_verification', methods=['POST'])
@app.route('/execute_with_verification', methods=['POST'])
def execute_command_with_verification():
    """Execute command and verify the result based on provided verification criteria"""
    data = request.json
    shell = data.get('shell', False)
    command = data.get('command', "" if shell else [])
    verification = data.get('verification', {})
    max_wait_time = data.get('max_wait_time', 10)  # Maximum wait time in seconds
    check_interval = data.get('check_interval', 1)  # Check interval in seconds

    if isinstance(command, str) and not shell:
        command = shlex.split(command)

    # Expand user directory
    for i, arg in enumerate(command):
        if arg.startswith("~/"):
            command[i] = os.path.expanduser(arg)

    # Execute the main command
    try:
        if platform_name == "Windows":
            flags = subprocess.CREATE_NO_WINDOW
        else:
            flags = 0
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            text=True,
            timeout=120,
            creationflags=flags,
        )

        # If no verification is needed, return immediately
        if not verification:
            return jsonify({
                'status': 'success',
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode
            })

        # Wait and verify the result
        import time
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            verification_passed = True

            # Check window existence if specified
            if 'window_exists' in verification:
                window_name = verification['window_exists']
                try:
                    if platform_name == 'Linux':
                        wmctrl_result = subprocess.run(['wmctrl', '-l'],
                                                     capture_output=True, text=True, check=True)
                        if window_name.lower() not in wmctrl_result.stdout.lower():
                            verification_passed = False
                    elif platform_name in ['Windows', 'Darwin']:
                        import pygetwindow as gw
                        windows = gw.getWindowsWithTitle(window_name)
                        if not windows:
                            verification_passed = False
                except Exception:
                    verification_passed = False

            # Check command execution if specified
            if 'command_success' in verification:
                verify_cmd = verification['command_success']
                try:
                    verify_result = subprocess.run(verify_cmd, shell=True,
                                                 capture_output=True, text=True, timeout=5)
                    if verify_result.returncode != 0:
                        verification_passed = False
                except Exception:
                    verification_passed = False

            if verification_passed:
                return jsonify({
                    'status': 'success',
                    'output': result.stdout,
                    'error': result.stderr,
                    'returncode': result.returncode,
                    'verification': 'passed',
                    'wait_time': time.time() - start_time
                })

            time.sleep(check_interval)

        # Verification failed
        return jsonify({
            'status': 'verification_failed',
            'output': result.stdout,
            'error': result.stderr,
            'returncode': result.returncode,
            'verification': 'failed',
            'wait_time': max_wait_time
        }), 500

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def _get_machine_architecture() -> str:
    """ Get the machine architecture, e.g., x86_64, arm64, aarch64, i386, etc.
    """
    architecture = platform.machine().lower()
    if architecture in ['amd32', 'amd64', 'x86', 'x86_64', 'x86-64', 'x64', 'i386', 'i686']:
        return 'amd'
    elif architecture in ['arm64', 'aarch64', 'aarch32']:
        return 'arm'
    else:
        return 'unknown'


@app.route('/setup/launch', methods=["POST"])
def launch_app():
    data = request.json
    shell = data.get("shell", False)
    command: List[str] = data.get("command", "" if shell else [])

    if isinstance(command, str) and not shell:
        command = shlex.split(command)

    # Expand user directory
    for i, arg in enumerate(command):
        if arg.startswith("~/"):
            command[i] = os.path.expanduser(arg)

    try:
        if 'google-chrome' in command and _get_machine_architecture() == 'arm':
            index = command.index('google-chrome')
            command[index] = 'chromium'  # arm64 chrome is not available yet, can only use chromium
        subprocess.Popen(command, shell=shell)
        return "{:} launched successfully".format(command if shell else " ".join(command))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/version', methods=['GET'])
def get_version():
    """Return server version to verify code updates."""
    return jsonify({
        "version": SERVER_VERSION,
        "features": [
            "panel_text_extraction",  # Extract text from presentation placeholders
        ]
    })


@app.route('/screenshot', methods=['GET'])
def capture_screen_with_cursor():
    # fixme: when running on virtual machines, the cursor is not captured, don't know why

    file_path = os.path.join(os.path.dirname(__file__), "screenshots", "screenshot.png")
    user_platform = platform.system()

    # Ensure the screenshots directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # fixme: This is a temporary fix for the cursor not being captured on Windows and Linux
    if user_platform == "Windows":
        def get_cursor():
            hcursor = win32gui.GetCursorInfo()[1]
            hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(0))
            hbmp = win32ui.CreateBitmap()
            hbmp.CreateCompatibleBitmap(hdc, 36, 36)
            hdc = hdc.CreateCompatibleDC()
            hdc.SelectObject(hbmp)
            hdc.DrawIcon((0,0), hcursor)

            bmpinfo = hbmp.GetInfo()
            bmpstr = hbmp.GetBitmapBits(True)
            cursor = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1).convert("RGBA")

            win32gui.DestroyIcon(hcursor)
            win32gui.DeleteObject(hbmp.GetHandle())
            hdc.DeleteDC()

            pixdata = cursor.load()

            width, height = cursor.size
            for y in range(height):
                for x in range(width):
                    if pixdata[x, y] == (0, 0, 0, 255):
                        pixdata[x, y] = (0, 0, 0, 0)

            hotspot = win32gui.GetIconInfo(hcursor)[1:3]

            return (cursor, hotspot)

        ratio = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100

        img = ImageGrab.grab(bbox=None, include_layered_windows=True)

        try:
            cursor, (hotspotx, hotspoty) = get_cursor()

            pos_win = win32gui.GetCursorPos()
            pos = (round(pos_win[0]*ratio - hotspotx), round(pos_win[1]*ratio - hotspoty))

            img.paste(cursor, pos, cursor)
        except Exception as e:
            logger.warning(f"Failed to capture cursor on Windows, screenshot will not have a cursor. Error: {e}")

        img.save(file_path)
        # Move mouse 1 pixel relative to current position to wake up monitor/GPU
        # We use moveRelNone to avoid triggering UI elements
        current_x, current_y = pyautogui.position()
        pyautogui.moveTo(current_x + 1, current_y)
        pyautogui.moveTo(current_x, current_y)
        # =====================================

        max_screenshot_attempts = 3
        for _screenshot_attempt in range(max_screenshot_attempts):
            try:
                cursor_obj = Xcursor()
                imgarray = cursor_obj.getCursorImageArrayFast()
                cursor_img = Image.fromarray(imgarray)

                # Taking screenshot after the wake-up
                screenshot = pyautogui.screenshot()

                cursor_x, cursor_y = pyautogui.position()
                screenshot.paste(cursor_img, (cursor_x, cursor_y), cursor_img)
                screenshot.save(file_path)
                break  # Success
            except Exception as e:
                logger.warning(f"Screenshot attempt {_screenshot_attempt + 1}/{max_screenshot_attempts} failed: {e}")
                # Clean up stale temp files that may cause PIL errors
                import glob
                for tmp_png in glob.glob("/tmp/tmp*.png"):
                    try:
                        os.remove(tmp_png)
                    except OSError:
                        pass
                if _screenshot_attempt == max_screenshot_attempts - 1:
                    logger.error(f"All {max_screenshot_attempts} screenshot attempts failed, returning error")
                    return jsonify({"status": "error", "message": f"Screenshot failed: {e}"}), 503
                time.sleep(0.5)
    elif user_platform == "Darwin":  # (Mac OS)
        # Use the screencapture utility to capture the screen with the cursor
        subprocess.run(["screencapture", "-C", file_path])
    else:
        logger.warning(f"The platform you're using ({user_platform}) is not currently supported")

    return send_file(file_path, mimetype='image/png')


def _has_active_terminal(desktop: Accessible) -> bool:
    """ A quick check whether the terminal window is open and active.
    """
    for app in desktop:
        if app.getRoleName() == "application" and app.name == "gnome-terminal-server":
            for frame in app:
                if frame.getRoleName() == "frame" and frame.getState().contains(pyatspi.STATE_ACTIVE):
                    return True
    return False


@app.route('/terminal', methods=['GET'])
def get_terminal_output():
    user_platform = platform.system()
    output: Optional[str] = None
    try:
        if user_platform == "Linux":
            desktop: Accessible = pyatspi.Registry.getDesktop(0)
            if _has_active_terminal(desktop):
                desktop_xml: _Element = _create_atspi_node(desktop)
                # 1. the terminal window (frame of application is st:active) is open and active
                # 2. the terminal tab (terminal status is st:focused) is focused
                xpath = '//application[@name="gnome-terminal-server"]/frame[@st:active="true"]//terminal[@st:focused="true"]'
                terminals: List[_Element] = desktop_xml.xpath(xpath, namespaces=_accessibility_ns_map_ubuntu)
                output = terminals[0].text.rstrip() if len(terminals) == 1 else None
        else:  # windows and macos platform is not implemented currently
            # raise NotImplementedError
            return "Currently not implemented for platform {:}.".format(platform.platform()), 500
        return jsonify({"output": output, "status": "success"})
    except Exception as e:
        logger.error("Failed to get terminal output. Error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


_accessibility_ns_map = {
    "ubuntu": {
        "st": "https://accessibility.ubuntu.example.org/ns/state",
        "attr": "https://accessibility.ubuntu.example.org/ns/attributes",
        "cp": "https://accessibility.ubuntu.example.org/ns/component",
        "doc": "https://accessibility.ubuntu.example.org/ns/document",
        "docattr": "https://accessibility.ubuntu.example.org/ns/document/attributes",
        "txt": "https://accessibility.ubuntu.example.org/ns/text",
        "val": "https://accessibility.ubuntu.example.org/ns/value",
        "act": "https://accessibility.ubuntu.example.org/ns/action",
    },
    "windows": {
        "st": "https://accessibility.windows.example.org/ns/state",
        "attr": "https://accessibility.windows.example.org/ns/attributes",
        "cp": "https://accessibility.windows.example.org/ns/component",
        "doc": "https://accessibility.windows.example.org/ns/document",
        "docattr": "https://accessibility.windows.example.org/ns/document/attributes",
        "txt": "https://accessibility.windows.example.org/ns/text",
        "val": "https://accessibility.windows.example.org/ns/value",
        "act": "https://accessibility.windows.example.org/ns/action",
        "class": "https://accessibility.windows.example.org/ns/class"
    },
    "macos": {
        "st": "https://accessibility.macos.example.org/ns/state",
        "attr": "https://accessibility.macos.example.org/ns/attributes",
        "cp": "https://accessibility.macos.example.org/ns/component",
        "doc": "https://accessibility.macos.example.org/ns/document",
        "txt": "https://accessibility.macos.example.org/ns/text",
        "val": "https://accessibility.macos.example.org/ns/value",
        "act": "https://accessibility.macos.example.org/ns/action",
        "role": "https://accessibility.macos.example.org/ns/role",
    }

}

_accessibility_ns_map_ubuntu = _accessibility_ns_map['ubuntu']
_accessibility_ns_map_windows = _accessibility_ns_map['windows']
_accessibility_ns_map_macos = _accessibility_ns_map['macos']

# A11y tree getter for Ubuntu
libreoffice_version_tuple: Optional[Tuple[int, ...]] = None
MAX_DEPTH = 50
MAX_WIDTH = 1024
MAX_CALLS = 5000


def _get_libreoffice_version() -> Tuple[int, ...]:
    """Function to get the LibreOffice version as a tuple of integers."""
    result = subprocess.run("libreoffice --version", shell=True, text=True, stdout=subprocess.PIPE)
    version_str = result.stdout.split()[1]  # Assuming version is the second word in the command output
    return tuple(map(int, version_str.split(".")))


def _create_atspi_node(node: Accessible, depth: int = 0, flag: Optional[str] = None) -> _Element:
    node_name = node.name
    attribute_dict: Dict[str, Any] = {"name": node_name}

    #  States
    states: List[StateType] = node.getState().get_states()
    for st in states:
        state_name: str = StateType._enum_lookup[st]
        state_name: str = state_name.split("_", maxsplit=1)[1].lower()
        if len(state_name) == 0:
            continue
        attribute_dict["{{{:}}}{:}".format(_accessibility_ns_map_ubuntu["st"], state_name)] = "true"

    #  Attributes
    attributes: Dict[str, str] = node.get_attributes()
    for attribute_name, attribute_value in attributes.items():
        if len(attribute_name) == 0:
            continue
        attribute_dict["{{{:}}}{:}".format(_accessibility_ns_map_ubuntu["attr"], attribute_name)] = attribute_value

    #  Component
    if attribute_dict.get("{{{:}}}visible".format(_accessibility_ns_map_ubuntu["st"]), "false") == "true" \
            and attribute_dict.get("{{{:}}}showing".format(_accessibility_ns_map_ubuntu["st"]), "false") == "true":
        try:
            component: Component = node.queryComponent()
        except NotImplementedError:
            pass
        else:
            bbox: Sequence[int] = component.getExtents(pyatspi.XY_SCREEN)
            attribute_dict["{{{:}}}screencoord".format(_accessibility_ns_map_ubuntu["cp"])] = \
                str(tuple(bbox[0:2]))
            attribute_dict["{{{:}}}size".format(_accessibility_ns_map_ubuntu["cp"])] = str(tuple(bbox[2:]))

    text = ""
    #  Text
    try:
        text_obj: ATText = node.queryText()
        # only text shown on current screen is available
        # attribute_dict["txt:text"] = text_obj.getText(0, text_obj.characterCount)
        text: str = text_obj.getText(0, text_obj.characterCount)
        # if flag=="thunderbird":
        # appeared in thunderbird (uFFFC) (not only in thunderbird), "Object
        # Replacement Character" in Unicode, "used as placeholder in text for
        # an otherwise unspecified object; uFFFD is another "Replacement
        # Character", just in case
        text = text.replace("\ufffc", "").replace("\ufffd", "")
    except NotImplementedError:
        pass

    #  Image, Selection, Value, Action
    try:
        node.queryImage()
        attribute_dict["image"] = "true"
    except NotImplementedError:
        pass

    try:
        node.querySelection()
        attribute_dict["selection"] = "true"
    except NotImplementedError:
        pass

    try:
        value: ATValue = node.queryValue()
        value_key = f"{{{_accessibility_ns_map_ubuntu['val']}}}"

        for attr_name, attr_func in [
            ("value", lambda: value.currentValue),
            ("min", lambda: value.minimumValue),
            ("max", lambda: value.maximumValue),
            ("step", lambda: value.minimumIncrement)
        ]:
            try:
                attribute_dict[f"{value_key}{attr_name}"] = str(attr_func())
            except:
                pass
    except NotImplementedError:
        pass

    try:
        action: ATAction = node.queryAction()
        for i in range(action.nActions):
            action_name: str = action.getName(i).replace(" ", "-")
            attribute_dict[
                "{{{:}}}{:}_desc".format(_accessibility_ns_map_ubuntu["act"], action_name)] = action.getDescription(
                i)
            attribute_dict[
                "{{{:}}}{:}_kb".format(_accessibility_ns_map_ubuntu["act"], action_name)] = action.getKeyBinding(i)
    except NotImplementedError:
        pass

    # Add from here if we need more attributes in the future...

    raw_role_name: str = node.getRoleName().strip()
    node_role_name = (raw_role_name or "unknown").replace(" ", "-")

    if not flag:
        if raw_role_name == "document spreadsheet":
            flag = "calc"
        if raw_role_name == "application" and node.name == "Thunderbird":
            flag = "thunderbird"

    xml_node = lxml.etree.Element(
        node_role_name,
        attrib=attribute_dict,
        nsmap=_accessibility_ns_map_ubuntu
    )

    if len(text) > 0:
        xml_node.text = text

    if depth == MAX_DEPTH:
        logger.warning("Max depth reached")
        return xml_node

    if flag == "calc" and node_role_name == "table":
        # Maximum column: 1024 if ver<=7.3 else 16384
        # Maximum row: 104 8576
        # Maximun sheet: 1 0000

        global libreoffice_version_tuple
        MAXIMUN_COLUMN = 1024 if libreoffice_version_tuple < (7, 4) else 16384
        MAX_ROW = 104_8576

        index_base = 0
        first_showing = False
        column_base = None
        for r in range(MAX_ROW):
            for clm in range(column_base or 0, MAXIMUN_COLUMN):
                child_node: Accessible = node[index_base + clm]
                showing: bool = child_node.getState().contains(STATE_SHOWING)
                if showing:
                    child_node: _Element = _create_atspi_node(child_node, depth + 1, flag)
                    if not first_showing:
                        column_base = clm
                        first_showing = True
                    xml_node.append(child_node)
                elif first_showing and column_base is not None or clm >= 500:
                    break
            if first_showing and clm == column_base or not first_showing and r >= 500:
                break
            index_base += MAXIMUN_COLUMN
        return xml_node
    else:
        try:
            for i, ch in enumerate(node):
                if i == MAX_WIDTH:
                    logger.warning("Max width reached")
                    break
                xml_node.append(_create_atspi_node(ch, depth + 1, flag))
        except:
            logger.warning("Error occurred during children traversing. Has Ignored. Node: %s",
                           lxml.etree.tostring(xml_node, encoding="unicode"))
        return xml_node


# A11y tree getter for Windows
def _create_pywinauto_node(node, nodes, depth: int = 0, flag: Optional[str] = None) -> _Element:
    nodes = nodes or set()
    if node in nodes:
        return
    nodes.add(node)

    attribute_dict: Dict[str, Any] = {"name": node.element_info.name}

    base_properties = {}
    try:
        base_properties.update(
            node.get_properties())  # get all writable/not writable properties, but have bugs when landing on chrome and it's slower!
    except:
        logger.debug("Failed to call get_properties(), trying to get writable properites")
        try:
            _element_class = node.__class__

            class TempElement(node.__class__):
                writable_props = pywinauto.base_wrapper.BaseWrapper.writable_props

            # Instantiate the subclass
            node.__class__ = TempElement
            # Retrieve properties using get_properties()
            properties = node.get_properties()
            node.__class__ = _element_class

            base_properties.update(properties)  # only get all writable properties
            logger.debug("get writable properties")
        except Exception as e:
            logger.error(e)
            pass

    # Count-cnt
    for attr_name in ["control_count", "button_count", "item_count", "column_count"]:
        try:
            attribute_dict[f"{{{_accessibility_ns_map_windows['cnt']}}}{attr_name}"] = base_properties[
                attr_name].lower()
        except:
            pass

    # Columns-cols
    try:
        attribute_dict[f"{{{_accessibility_ns_map_windows['cols']}}}columns"] = base_properties["columns"].lower()
    except:
        pass

    # Id-id
    for attr_name in ["control_id", "automation_id", "window_id"]:
        try:
            attribute_dict[f"{{{_accessibility_ns_map_windows['id']}}}{attr_name}"] = base_properties[attr_name].lower()
        except:
            pass

    #  States
    # 19 sec out of 20
    for attr_name, attr_func in [
        ("enabled", lambda: node.is_enabled()),
        ("visible", lambda: node.is_visible()),
        # ("active", lambda: node.is_active()), # occupied most of the time: 20s out of 21s for slack, 51.5s out of 54s for WeChat # maybe use for cutting branches
        ("minimized", lambda: node.is_minimized()),
        ("maximized", lambda: node.is_maximized()),
        ("normal", lambda: node.is_normal()),
        ("unicode", lambda: node.is_unicode()),
        ("collapsed", lambda: node.is_collapsed()),
        ("checkable", lambda: node.is_checkable()),
        ("checked", lambda: node.is_checked()),
        ("focused", lambda: node.is_focused()),
        ("keyboard_focused", lambda: node.is_keyboard_focused()),
        ("selected", lambda: node.is_selected()),
        ("selection_required", lambda: node.is_selection_required()),
        ("pressable", lambda: node.is_pressable()),
        ("pressed", lambda: node.is_pressed()),
        ("expanded", lambda: node.is_expanded()),
        ("editable", lambda: node.is_editable()),
        ("has_keyboard_focus", lambda: node.has_keyboard_focus()),
        ("is_keyboard_focusable", lambda: node.is_keyboard_focusable()),
    ]:
        try:
            attribute_dict[f"{{{_accessibility_ns_map_windows['st']}}}{attr_name}"] = str(attr_func()).lower()
        except:
            pass

    #  Component
    try:
        rectangle = node.rectangle()
        attribute_dict["{{{:}}}screencoord".format(_accessibility_ns_map_windows["cp"])] = \
            "({:d}, {:d})".format(rectangle.left, rectangle.top)
        attribute_dict["{{{:}}}size".format(_accessibility_ns_map_windows["cp"])] = \
            "({:d}, {:d})".format(rectangle.width(), rectangle.height())

    except Exception as e:
        logger.error("Error accessing rectangle: ", e)

    #  Text
    text: str = node.window_text()
    if text == attribute_dict["name"]:
        text = ""

    #  Selection
    if hasattr(node, "select"):
        attribute_dict["selection"] = "true"

    # Value
    for attr_name, attr_funcs in [
        ("step", [lambda: node.get_step()]),
        ("value", [lambda: node.value(), lambda: node.get_value(), lambda: node.get_position()]),
        ("min", [lambda: node.min_value(), lambda: node.get_range_min()]),
        ("max", [lambda: node.max_value(), lambda: node.get_range_max()])
    ]:
        for attr_func in attr_funcs:
            if hasattr(node, attr_func.__name__):
                try:
                    attribute_dict[f"{{{_accessibility_ns_map_windows['val']}}}{attr_name}"] = str(attr_func())
                    break  # exit once the attribute is set successfully
                except:
                    pass

    attribute_dict["{{{:}}}class".format(_accessibility_ns_map_windows["class"])] = str(type(node))

    # class_name
    for attr_name in ["class_name", "friendly_class_name"]:
        try:
            attribute_dict[f"{{{_accessibility_ns_map_windows['class']}}}{attr_name}"] = base_properties[
                attr_name].lower()
        except:
            pass

    node_role_name: str = node.class_name().lower().replace(" ", "-")
    node_role_name = "".join(
        map(lambda _ch: _ch if _ch.isidentifier() or _ch in {"-"} or _ch.isalnum() else "-", node_role_name))

    if node_role_name.strip() == "":
        node_role_name = "unknown"
    if not node_role_name[0].isalpha():
        node_role_name = "tag" + node_role_name

    xml_node = lxml.etree.Element(
        node_role_name,
        attrib=attribute_dict,
        nsmap=_accessibility_ns_map_windows
    )

    if text is not None and len(text) > 0 and text != attribute_dict["name"]:
        xml_node.text = text

    if depth == MAX_DEPTH:
        logger.warning("Max depth reached")
        return xml_node

    # use multi thread to accelerate children fetching
    children = node.children()
    if children:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_child = [executor.submit(_create_pywinauto_node, ch, nodes, depth + 1, flag) for ch in
                               children[:MAX_WIDTH]]
        try:
            xml_node.extend([future.result() for future in concurrent.futures.as_completed(future_to_child)])
        except Exception as e:
            logger.error(f"Exception occurred: {e}")
    return xml_node


# A11y tree getter for macOS

def _create_axui_node(node, nodes: set = None, depth: int = 0, bbox: tuple = None):
    nodes = nodes or set()
    if node in nodes:
        return
    nodes.add(node)

    reserved_keys = {
        "AXEnabled": "st",
        "AXFocused": "st",
        "AXFullScreen": "st",
        "AXTitle": "attr",
        "AXChildrenInNavigationOrder": "attr",
        "AXChildren": "attr",
        "AXFrame": "attr",
        "AXRole": "role",
        "AXHelp": "attr",
        "AXRoleDescription": "role",
        "AXSubrole": "role",
        "AXURL": "attr",
        "AXValue": "val",
        "AXDescription": "attr",
        "AXDOMIdentifier": "attr",
        "AXSelected": "st",
        "AXInvalid": "st",
        "AXRows": "attr",
        "AXColumns": "attr",
    }
    attribute_dict = {}

    if depth == 0:
        bbox = (
            node["kCGWindowBounds"]["X"],
            node["kCGWindowBounds"]["Y"],
            node["kCGWindowBounds"]["X"] + node["kCGWindowBounds"]["Width"],
            node["kCGWindowBounds"]["Y"] + node["kCGWindowBounds"]["Height"]
        )
        app_ref = ApplicationServices.AXUIElementCreateApplication(node["kCGWindowOwnerPID"])

        attribute_dict["name"] = node["kCGWindowOwnerName"]
        if attribute_dict["name"] != "Dock":
            error_code, app_wins_ref = ApplicationServices.AXUIElementCopyAttributeValue(
                app_ref, "AXWindows", None)
            if error_code:
                logger.error("MacOS parsing %s encountered Error code: %d", app_ref, error_code)
        else:
            app_wins_ref = [app_ref]
        node = app_wins_ref[0]

    error_code, attr_names = ApplicationServices.AXUIElementCopyAttributeNames(node, None)

    if error_code:
        # -25202: AXError.invalidUIElement
        #         The accessibility object received in this event is invalid.
        return

    value = None

    if "AXFrame" in attr_names:
        error_code, attr_val = ApplicationServices.AXUIElementCopyAttributeValue(node, "AXFrame", None)
        rep = repr(attr_val)
        x_value = re.search(r"x:(-?[\d.]+)", rep)
        y_value = re.search(r"y:(-?[\d.]+)", rep)
        w_value = re.search(r"w:(-?[\d.]+)", rep)
        h_value = re.search(r"h:(-?[\d.]+)", rep)
        type_value = re.search(r"type\s?=\s?(\w+)", rep)
        value = {
            "x": float(x_value.group(1)) if x_value else None,
            "y": float(y_value.group(1)) if y_value else None,
            "w": float(w_value.group(1)) if w_value else None,
            "h": float(h_value.group(1)) if h_value else None,
            "type": type_value.group(1) if type_value else None,
        }

        if not any(v is None for v in value.values()):
            x_min = max(bbox[0], value["x"])
            x_max = min(bbox[2], value["x"] + value["w"])
            y_min = max(bbox[1], value["y"])
            y_max = min(bbox[3], value["y"] + value["h"])

            if x_min > x_max or y_min > y_max:
                # No intersection
                return

    role = None
    text = None

    for attr_name, ns_key in reserved_keys.items():
        if attr_name not in attr_names:
            continue

        if value and attr_name == "AXFrame":
            bb = value
            if not any(v is None for v in bb.values()):
                attribute_dict["{{{:}}}screencoord".format(_accessibility_ns_map_macos["cp"])] = \
                    "({:d}, {:d})".format(int(bb["x"]), int(bb["y"]))
                attribute_dict["{{{:}}}size".format(_accessibility_ns_map_macos["cp"])] = \
                    "({:d}, {:d})".format(int(bb["w"]), int(bb["h"]))
            continue

        error_code, attr_val = ApplicationServices.AXUIElementCopyAttributeValue(node, attr_name, None)

        full_attr_name = f"{{{_accessibility_ns_map_macos[ns_key]}}}{attr_name}"

        if attr_name == "AXValue" and not text:
            text = str(attr_val)
            continue

        if attr_name == "AXRoleDescription":
            role = attr_val
            continue

        # Set the attribute_dict
        if not (isinstance(attr_val, ApplicationServices.AXUIElementRef)
                or isinstance(attr_val, (AppKit.NSArray, list))):
            if attr_val is not None:
                attribute_dict[full_attr_name] = str(attr_val)

    node_role_name = role.lower().replace(" ", "_") if role else "unknown_role"

    xml_node = lxml.etree.Element(
        node_role_name,
        attrib=attribute_dict,
        nsmap=_accessibility_ns_map_macos
    )

    if text is not None and len(text) > 0:
        xml_node.text = text

    if depth == MAX_DEPTH:
        logger.warning("Max depth reached")
        return xml_node

    future_to_child = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for attr_name, ns_key in reserved_keys.items():
            if attr_name not in attr_names:
                continue

            error_code, attr_val = ApplicationServices.AXUIElementCopyAttributeValue(node, attr_name, None)
            if isinstance(attr_val, ApplicationServices.AXUIElementRef):
                future_to_child.append(executor.submit(_create_axui_node, attr_val, nodes, depth + 1, bbox))

            elif isinstance(attr_val, (AppKit.NSArray, list)):
                for child in attr_val:
                    future_to_child.append(executor.submit(_create_axui_node, child, nodes, depth + 1, bbox))

        try:
            for future in concurrent.futures.as_completed(future_to_child):
                result = future.result()
                if result is not None:
                    xml_node.append(result)
        except Exception as e:
            logger.error(f"Exception occurred: {e}")

    return xml_node


@app.route("/accessibility", methods=["GET"])
def get_accessibility_tree():
    os_name: str = platform.system()

    # AT-SPI works for KDE as well
    if os_name == "Linux":
        output = get_accessibility_tree_nested()
        return jsonify({"AT": output})
        # global libreoffice_version_tuple
        # libreoffice_version_tuple = _get_libreoffice_version()

        # desktop: Accessible = pyatspi.Registry.getDesktop(0)
        # xml_node = lxml.etree.Element("desktop-frame", nsmap=_accessibility_ns_map_ubuntu)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(_create_atspi_node, app_node, 2) for app_node in desktop]
        #     for future in concurrent.futures.as_completed(futures):
        #         xml_tree = future.result()
        #         xml_node.append(xml_tree)
        # return jsonify({"AT": lxml.etree.tostring(xml_node, encoding="unicode")})

    elif os_name == "Windows":
        # Attention: Windows a11y tree is implemented to be read through `pywinauto` module, however,
        # two different backends `win32` and `uia` are supported and different results may be returned
        desktop: Desktop = Desktop(backend="uia")
        xml_node = lxml.etree.Element("desktop", nsmap=_accessibility_ns_map_windows)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_create_pywinauto_node, wnd, {}, 1) for wnd in desktop.windows()]
            for future in concurrent.futures.as_completed(futures):
                xml_tree = future.result()
                xml_node.append(xml_tree)
        return jsonify({"AT": lxml.etree.tostring(xml_node, encoding="unicode")})

    elif os_name == "Darwin":
        # TODO: Add Dock and MenuBar
        xml_node = lxml.etree.Element("desktop", nsmap=_accessibility_ns_map_macos)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            foreground_windows = [
                win for win in Quartz.CGWindowListCopyWindowInfo(
                    (Quartz.kCGWindowListExcludeDesktopElements |
                     Quartz.kCGWindowListOptionOnScreenOnly),
                    Quartz.kCGNullWindowID
                ) if win["kCGWindowLayer"] == 0 and win["kCGWindowOwnerName"] != "Window Server"
            ]
            dock_info = [
                win for win in Quartz.CGWindowListCopyWindowInfo(
                    Quartz.kCGWindowListOptionAll,
                    Quartz.kCGNullWindowID
                ) if win.get("kCGWindowName", None) == "Dock"
            ]

            futures = [
                executor.submit(_create_axui_node, wnd, None, 0)
                for wnd in foreground_windows + dock_info
            ]

            for future in concurrent.futures.as_completed(futures):
                xml_tree = future.result()
                if xml_tree is not None:
                    xml_node.append(xml_tree)

        return jsonify({"AT": lxml.etree.tostring(xml_node, encoding="unicode")})

    else:
        return "Currently not implemented for platform {:}.".format(platform.platform()), 500


# ---------- Minimal/Clean Accessibility API (Linux only) ----------

def _is_showing_minimal(node: Accessible) -> bool:
    """True if node is visible and showing on screen according to AT-SPI state."""
    try:
        st = node.getState()
    except Exception:
        return False
    return (st.contains(pyatspi.STATE_VISIBLE)
            and st.contains(pyatspi.STATE_SHOWING))


def _is_menu_expanded(menu_node: Accessible) -> bool:
    """Check if a menu is expanded by checking if it's selected in its parent's selection."""
    try:
        parent = menu_node.parent
        if parent:
            parent_role = (parent.getRoleName() or "").strip().lower()
            if parent_role in ("menu bar", "menu"):
                sel = parent.querySelection()
                if sel and sel.nSelectedChildren > 0:
                    for i in range(sel.nSelectedChildren):
                        selected = sel.getSelectedChild(i)
                        if selected and selected == menu_node:
                            return True
    except Exception:
        pass
    return False


def _get_bounds_minimal(node: Accessible):
    """Get absolute screen-space bounds (x, y, w, h) of the node, or None."""
    try:
        comp: Component = node.queryComponent()
    except NotImplementedError:
        return None
    except Exception:
        return None

    try:
        x, y, w, h = comp.getExtents(0)  # 0 == screen coordinates
    except Exception:
        return None

    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def _get_actions_minimal(node: Accessible):
    """Return list of action names, or empty list if none."""
    try:
        act: ATAction = node.queryAction()
    except NotImplementedError:
        return []
    except Exception:
        return []

    names = []
    for i in range(act.nActions):
        try:
            names.append(act.getName(i))
        except Exception:
            pass
    return names


def _get_text_minimal(node: Accessible) -> str:
    """Get textual content if any, stripped."""
    try:
        txt: ATText = node.queryText()
    except NotImplementedError:
        return ""
    except Exception:
        return ""

    try:
        s = txt.getText(0, txt.characterCount)
        s = s.replace("\ufffc", "").replace("\ufffd", "")
        return s.strip()
    except Exception:
        return ""


def _get_text_selection(node: Accessible) -> dict:
    """Get text selection info if any."""
    try:
        txt: ATText = node.queryText()
        n_selections = txt.getNSelections()
        if n_selections > 0:
            start, end = txt.getSelection(0)
            if start >= 0 and end > start:
                selected_text = txt.getText(start, end)
                return {
                    "start": start,
                    "end": end,
                    "text": selected_text.strip()
                }
    except Exception:
        pass
    return None


def _get_window_context_minimal(node: Accessible):
    """Walk up ancestors to find the application name and window title."""
    app_name = None
    window_title = None
    cur = node
    while cur is not None:
        try:
            role = cur.getRoleName() or ""
        except Exception:
            role = ""
        name = getattr(cur, "name", None)

        if role == "application" and name:
            app_name = name
        if role in ("frame", "dialog", "window") and name:
            window_title = name

        try:
            cur = cur.parent
        except Exception:
            cur = None

    return app_name, window_title


def _collect_all_visible_elements_minimal(root: Accessible, max_depth: int = 40):
    """
    Traverse accessibility tree and collect *all* visible elements that have
    geometry (Component), including static labels and text.
    """
    elements = []

    def walk(node: Accessible, depth: int = 0):
        if depth > max_depth:
            return

        if not _is_showing_minimal(node):
            return

        bounds = _get_bounds_minimal(node)
        if bounds is not None:
            text = _get_text_minimal(node)
            name = (node.name or "").strip()
            role = (node.getRoleName() or "").strip().lower()
            actions = _get_actions_minimal(node)
            x, y, w, h = bounds
            cx = x + w // 2
            cy = y + h // 2
            app_name, window_title = _get_window_context_minimal(node)

            elements.append({
                "acc": node,  # keep for occlusion check
                "app": app_name,
                "window": window_title,
                "role": role,
                "name": name,
                "text": text,
                "bounds": {"x": x, "y": y, "w": w, "h": h},
                "center": {"x": cx, "y": cy},
                "actions": actions,
            })

        # Warmup: Access node attributes to trigger AT-SPI cache population (needed for Chrome)
        try:
            node.getState()
            node.get_attributes()
            node.queryComponent()
        except Exception:
            pass

        # Recurse into children
        # Use Python iterator instead of getChildAtIndex for better AT-SPI cache handling
        try:
            for child in node:
                if child:
                    walk(child, depth + 1)
        except Exception:
            return

    walk(root)
    return elements


def _element_visible_at_center_minimal(desktop_comp: Component, elem: dict) -> bool:
    """
    Use GetAccessibleAtPoint to decide whether elem is really visible/topmost
    at its intended click/anchor point (center).
    """
    cx = elem["center"]["x"]
    cy = elem["center"]["y"]

    try:
        hit = desktop_comp.getAccessibleAtPoint(cx, cy, pyatspi.DESKTOP_COORDS)
    except Exception:
        return False

    if hit is None:
        return False

    target = elem["acc"]

    # Walk up from 'hit' to see if 'target' is an ancestor
    cur = hit
    while cur is not None:
        if cur == target:
            return True
        try:
            cur = cur.parent
        except Exception:
            cur = None

    # Walk down from 'target' to see if 'hit' is a descendant
    stack = [target]
    while stack:
        n = stack.pop()
        if n == hit:
            return True
        # Use Python iterator instead of getChildAtIndex for better AT-SPI cache handling
        try:
            for ch in n:
                if ch:
                    stack.append(ch)
        except Exception:
            pass

    return False


def _filter_visible_elements_minimal(desktop: Accessible, elements: list):
    """Keep only elements that are not occluded at their center point."""
    try:
        desktop_comp: Component = desktop.queryComponent()
    except NotImplementedError:
        return []
    except Exception:
        return []

    visible = []
    for elem in elements:
        if _element_visible_at_center_minimal(desktop_comp, elem):
            cleaned = {k: v for k, v in elem.items() if k != "acc"}
            visible.append(cleaned)
    return visible


@app.route("/accessibility_minimal", methods=["GET"])
def get_accessibility_minimal():
    """
    Return a JSON snapshot of all visible AT-SPI elements on the current
    desktop, including nested labels and text, with absolute screen
    coordinates and occlusion handled via hit-testing.

    This is a cleaner, less noisy alternative to /accessibility that:
    - Returns flat JSON instead of verbose XML
    - Properly handles occlusion using GetAccessibleAtPoint (optional)
    - Only includes elements that are actually visible on screen

    Query parameters:
    - filter_occlusion: "true" (default) or "false" - whether to filter occluded elements
    """
    os_name: str = platform.system()

    if os_name != "Linux":
        return jsonify({"error": "accessibility_minimal is only implemented for Linux"}), 500

    # Check if occlusion filtering is requested (default: false for now since it's unreliable)
    filter_occlusion = request.args.get('filter_occlusion', 'false').lower() == 'true'

    try:
        desktop: Accessible = pyatspi.Registry.getDesktop(0)
    except Exception as e:
        return jsonify({"error": f"Failed to get desktop: {e}"}), 500

    # Screen size from desktop component extents
    try:
        desktop_comp: Component = desktop.queryComponent()
        dx, dy, dw, dh = desktop_comp.getExtents(0)
    except Exception:
        dx = dy = 0
        dw = dh = 0

    # Collect candidates from all apps under the desktop
    candidates = []
    try:
        for app_node in desktop:
            candidates.extend(_collect_all_visible_elements_minimal(app_node))
    except Exception as e:
        logger.error(f"Error during AT-SPI traversal: {e}")

    # Apply occlusion filtering if requested
    if filter_occlusion:
        visible = _filter_visible_elements_minimal(desktop, candidates)
    else:
        # Just remove the 'acc' key from all elements
        visible = [{k: v for k, v in elem.items() if k != "acc"} for elem in candidates]

    # If we don't have valid screen width/height, compute bounds from elements
    if (dw <= 0 or dh <= 0) and visible:
        min_x = min(e["bounds"]["x"] for e in visible)
        min_y = min(e["bounds"]["y"] for e in visible)
        max_x = max(e["bounds"]["x"] + e["bounds"]["w"] for e in visible)
        max_y = max(e["bounds"]["y"] + e["bounds"]["h"] for e in visible)
        dx, dy = min_x, min_y
        dw, dh = max_x - min_x, max_y - min_y

    payload = {
        "screen": {
            "x": dx,
            "y": dy,
            "width": dw,
            "height": dh,
        },
        "elements": visible,
        "filter_occlusion": filter_occlusion,
        "total_candidates": len(candidates),
    }

    return jsonify(payload)


def _get_window_stacking_order():
    """
    Get the stacking order of windows using X11's _NET_CLIENT_LIST_STACKING.
    Returns a list of window IDs from bottom to top (higher index = more on top).
    """
    try:
        d = display.Display()
        root = d.screen().root
        # _NET_CLIENT_LIST_STACKING gives windows in stacking order (bottom to top)
        stacking_atom = d.intern_atom('_NET_CLIENT_LIST_STACKING')
        stacking_prop = root.get_full_property(stacking_atom, X.AnyPropertyType)
        if stacking_prop:
            return list(stacking_prop.value)
    except Exception:
        pass
    return []


def _get_active_window_id():
    """Get the currently active/focused window ID using X11."""
    try:
        d = display.Display()
        root = d.screen().root
        active_atom = d.intern_atom('_NET_ACTIVE_WINDOW')
        active_prop = root.get_full_property(active_atom, X.AnyPropertyType)
        if active_prop and active_prop.value:
            return active_prop.value[0]
    except Exception:
        pass
    return None


def _rect_intersection_area(r1, r2):
    """Calculate intersection area between two rectangles (x, y, w, h)."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    # Calculate intersection
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    if ix1 < ix2 and iy1 < iy2:
        return (ix2 - ix1) * (iy2 - iy1)
    return 0


def _is_window_occluded(window_bounds, windows_above, threshold: float = 0.0):
    """
    Check if a window is significantly occluded by windows above it.

    Args:
        window_bounds: (x, y, w, h) of the window to check
        windows_above: List of (x, y, w, h) bounds of windows above this one
        threshold: Occlusion threshold (0.0 = center-based, 0.9 = 90% area covered)

    Returns True if the window should be considered occluded.
    """
    x, y, w, h = window_bounds
    window_area = w * h

    if threshold <= 0:
        # Center-based occlusion: check if center is covered
        cx, cy = x + w // 2, y + h // 2
        for ax, ay, aw, ah in windows_above:
            if ax <= cx < ax + aw and ay <= cy < ay + ah:
                return True
        return False
    else:
        # Area-based occlusion: check if enough area is covered
        total_occluded = 0
        for above_bounds in windows_above:
            intersection = _rect_intersection_area(window_bounds, above_bounds)
            total_occluded += intersection

        if window_area > 0:
            occlusion_ratio = total_occluded / window_area
            return occlusion_ratio >= threshold
        return False


@app.route("/accessibility_tree", methods=["GET"])
def get_accessibility_tree_nested():
    """
    Return a nested DOM-like tree structure of all visible AT-SPI elements.

    This endpoint returns the accessibility tree in a hierarchical structure
    where each element contains its children, similar to a DOM tree.

    Window occlusion is handled by:
    1. Determining window stacking order (active window on top)
    2. Filtering out windows based on occlusion mode

    Query parameters:
    - max_depth: Maximum depth to traverse (default: 40)
    - filter_occluded: Whether to filter occluded windows (default: true)
    - occlusion_mode: How to detect occlusion (default: "center")
        - "center": Filter if window center is covered (strict)
        - "area": Filter if 90% of window area is covered (relaxed)
        - "active_only": Only show the active/focused window
    - flat: If "true", return only actionable elements in a flat list (default: false)
    """
    os_name: str = platform.system()

    if os_name != "Linux":
        return jsonify({"error": "accessibility_tree is only implemented for Linux"}), 500

    max_depth = int(request.args.get('max_depth', '40'))
    filter_occluded = request.args.get('filter_occluded', 'true').lower() == 'true'
    occlusion_mode = request.args.get('occlusion_mode', 'center').lower()
    flat_mode = request.args.get('flat', 'true').lower() == 'true'

    try:
        desktop: Accessible = pyatspi.Registry.getDesktop(0)
    except Exception as e:
        return jsonify({"error": f"Failed to get desktop: {e}"}), 500

    # Screen size from desktop component extents
    try:
        desktop_comp: Component = desktop.queryComponent()
        dx, dy, dw, dh = desktop_comp.getExtents(0)
    except Exception:
        dx = dy = 0
        dw = dh = 1920  # Default fallback

    # Get window stacking order and active window for occlusion filtering
    stacking_order = _get_window_stacking_order() if filter_occluded else []
    active_window_id = _get_active_window_id() if filter_occluded else None

    # Collect all top-level windows (frames/dialogs) with their bounds and app info
    # We'll use this to determine occlusion
    top_level_windows = []  # List of (app_node, frame_node, bounds, window_name)

    # Background service apps that should be completely excluded
    # These don't have visible UI that users interact with
    BACKGROUND_APPS = {'ibus-x11', 'ibus-extension-gtk3', 'gsd-color', 'gsd-keyboard',
                       'gsd-media-keys', 'gsd-wacom', 'gsd-power', 'gsd-xsettings',
                       'evolution-alarm-notify', 'xdg-desktop-portal-gtk', 'gnome-calendar',
                       'gnome-software'}

    # Apps whose windows should not be counted for occlusion (they're overlays/compositors)
    # but their UI elements (dock, panel) should still be included in the tree
    OVERLAY_APPS = {'gnome-shell', 'gjs'}

    def collect_top_level_windows(app_node):
        """Collect all frame/dialog windows from an application for occlusion calculation."""
        app_name = (app_node.name or "").strip().lower()

        # Skip background service apps entirely
        if app_name in BACKGROUND_APPS:
            return []

        # Skip overlay apps from occlusion calculation (but they'll still be in the tree)
        if app_name in OVERLAY_APPS:
            return []

        windows = []
        # Warmup: Access app attributes to trigger AT-SPI cache population (needed for Chrome)
        try:
            app_node.getState()
            app_node.get_attributes()
        except Exception:
            pass

        # Use Python iterator instead of getChildAtIndex for better AT-SPI cache handling
        try:
            for child in app_node:
                if child:
                    # Warmup child as well
                    try:
                        child.getState()
                        child.get_attributes()
                        child.queryComponent()
                    except Exception:
                        pass

                    role = (child.getRoleName() or "").strip().lower()
                    if role in ("frame", "dialog", "window", "alert", "file chooser"):
                        bounds = _get_bounds_minimal(child)
                        if bounds:
                            name = (child.name or "").strip()
                            windows.append((app_node, child, bounds, name))
        except Exception:
            pass
        return windows

    # First pass: collect all top-level windows
    try:
        for app_node in desktop:
            windows = collect_top_level_windows(app_node)
            top_level_windows.extend(windows)
    except Exception:
        pass

    # Build a map from window class/name to X11 stacking index
    x11_stacking_map = {}  # Maps (class_name, window_name) to stacking index
    try:
        d = display.Display()
        stacking_atom = d.intern_atom('_NET_CLIENT_LIST_STACKING')
        stacking_prop = d.screen().root.get_full_property(stacking_atom, X.AnyPropertyType)
        if stacking_prop:
            for idx, wid in enumerate(stacking_prop.value):
                try:
                    win = d.create_resource_object('window', wid)
                    wm_name = win.get_wm_name() or ""
                    wm_class = win.get_wm_class()
                    class_name = wm_class[1].lower() if wm_class else ""
                    # Store both class and name for matching
                    x11_stacking_map[(class_name, wm_name.lower()[:50])] = idx
                    x11_stacking_map[(class_name, "")] = idx  # Also store by class only
                except Exception:
                    pass
    except Exception:
        pass

    # Sort windows by X11 stacking order
    def normalize_name(name):
        """Normalize app/window names for comparison."""
        return name.lower().replace(" ", "").replace("-", "").replace("_", "")

    def get_stacking_index(window_info):
        app_node, frame_node, bounds, win_name = window_info
        app_name = (app_node.name or "").strip()
        app_name_norm = normalize_name(app_name)
        win_name_lower = (win_name or "").lower()[:50]
        win_name_norm = normalize_name(win_name or "")
        role = (frame_node.getRoleName() or "").strip().lower()

        # Try to find matching X11 window
        best_match_idx = -1

        for (cls, name), idx in x11_stacking_map.items():
            cls_norm = normalize_name(cls)

            # Check if app names match (normalized)
            if cls_norm and app_name_norm and (cls_norm in app_name_norm or app_name_norm in cls_norm):
                # If we have a window name match too, this is a strong match
                if name and win_name_lower and (name in win_name_lower or win_name_lower in name):
                    return idx
                # Otherwise, remember this as a potential match
                if idx > best_match_idx:
                    best_match_idx = idx

            # Also try matching by window title (for apps like soffice -> libreoffice-calc)
            # Match if X11 window name matches AT-SPI window title
            elif name and win_name_lower and (name in win_name_lower or win_name_lower in name):
                if idx > best_match_idx:
                    best_match_idx = idx
            # Or if X11 class name appears in AT-SPI window title (e.g., "LibreOffice Calc" in title)
            elif cls_norm and win_name_norm and (cls_norm in win_name_norm or win_name_norm in cls_norm):
                if idx > best_match_idx:
                    best_match_idx = idx

        if best_match_idx >= 0:
            return best_match_idx

        # Dialogs that are ACTIVE should be on top of their parent app
        # but not necessarily above other apps
        if role == "dialog":
            try:
                state = frame_node.getState()
                if state.contains(pyatspi.STATE_ACTIVE):
                    # Find parent app's stacking index and add a small offset
                    for (cls, _), idx in x11_stacking_map.items():
                        cls_norm = normalize_name(cls)
                        if cls_norm and app_name_norm and (cls_norm in app_name_norm or app_name_norm in cls_norm):
                            return idx + 0.5  # Dialog is above its parent but below next app
            except Exception:
                pass

        # Fallback: use AT-SPI state for windows we couldn't match
        try:
            state = frame_node.getState()
            if state.contains(pyatspi.STATE_ACTIVE):
                return 999999  # Active window is on top
            if state.contains(pyatspi.STATE_FOCUSED):
                return 999998  # Focused window is near top
        except Exception:
            pass
        return -1  # Unknown windows go to bottom

    top_level_windows.sort(key=get_stacking_index)

    # Check for modal dialogs - if a modal dialog is active, only show it
    # Modal dialogs block interaction with other windows
    modal_dialog = None
    modal_dialog_app = None
    for app_node, frame_node, bounds, name in top_level_windows:
        try:
            role = (frame_node.getRoleName() or "").strip().lower()
            state = frame_node.getState()
            if state.contains(pyatspi.STATE_MODAL) and state.contains(pyatspi.STATE_SHOWING):
                modal_dialog = frame_node
                modal_dialog_app = app_node
                break
        except Exception:
            pass

    # Also check for file chooser dialogs which are modal
    if not modal_dialog:
        for app_node, frame_node, bounds, name in top_level_windows:
            try:
                role = (frame_node.getRoleName() or "").strip().lower()
                if role in ("file chooser", "dialog", "alert"):
                    state = frame_node.getState()
                    if state.contains(pyatspi.STATE_ACTIVE) and state.contains(pyatspi.STATE_SHOWING):
                        # Check if it's a file chooser or has modal-like behavior
                        if role == "file chooser" or state.contains(pyatspi.STATE_MODAL):
                            modal_dialog = frame_node
                            modal_dialog_app = app_node
                            break
            except Exception:
                pass

    # Determine which windows are occluded based on mode
    visible_windows = set()  # Set of frame_node objects that are visible
    windows_above = []  # Accumulated bounds of windows processed (higher in stack)

    # If a modal dialog is active, only show that dialog
    if modal_dialog and filter_occluded:
        visible_windows.add(modal_dialog)
    else:
        # Set occlusion threshold based on mode
        if occlusion_mode == "area":
            occlusion_threshold = 0.9  # 90% coverage to be considered occluded
        else:
            occlusion_threshold = 0.0  # Center-based (default)

        # Process from top to bottom (reverse order)
        for app_node, frame_node, bounds, name in reversed(top_level_windows):
            if filter_occluded:
                if occlusion_mode == "active_only":
                    # Only show active/focused windows
                    try:
                        state = frame_node.getState()
                        is_active = state.contains(pyatspi.STATE_ACTIVE)
                        is_focused = state.contains(pyatspi.STATE_FOCUSED)
                        if not (is_active or is_focused):
                            continue
                    except Exception:
                        continue
                elif _is_window_occluded(bounds, windows_above, occlusion_threshold):
                    # This window is occluded by windows above
                    continue
            visible_windows.add(frame_node)
            windows_above.append(bounds)

    # LibreOffice Calc optimization constants
    # Maximum column: 1024 if ver<=7.3 else 16384
    # Maximum row: 1048576
    CALC_MAX_COLUMN = 16384  # Use newer LibreOffice limit
    CALC_MAX_ROW = 1048576

    def build_calc_table_children(table_node: Accessible, inherited_app: str = None, inherited_window: str = None) -> list:
        """
        Optimized traversal for LibreOffice Calc tables.
        Uses the table interface to correctly access cells by row/column coordinates.
        Only traverses visible cells instead of all columns × rows.
        """
        children = []

        try:
            table_iface = table_node.queryTable()
            n_rows = min(table_iface.nRows, CALC_MAX_ROW)
            n_cols = min(table_iface.nColumns, CALC_MAX_COLUMN)
        except Exception:
            # Fall back to old method if table interface not available
            return children

        first_showing_row = None
        last_showing_row = None
        first_showing_col = None
        last_showing_col = None

        # First pass: find the visible range by checking edges
        # Check first 100 rows to find visible range
        for row in range(min(100, n_rows)):
            for col in range(min(50, n_cols)):
                try:
                    cell = table_iface.getAccessibleAt(row, col)
                    if cell and cell.getState().contains(pyatspi.STATE_SHOWING):
                        if first_showing_row is None:
                            first_showing_row = row
                        last_showing_row = row
                        if first_showing_col is None or col < first_showing_col:
                            first_showing_col = col
                        if last_showing_col is None or col > last_showing_col:
                            last_showing_col = col
                except Exception:
                    pass

        # If no visible cells found, return empty
        if first_showing_row is None:
            return children

        # Second pass: collect all visible cells in the detected range
        # Add some buffer to catch all visible cells
        start_row = max(0, first_showing_row)
        end_row = min(n_rows, last_showing_row + 50)  # Buffer for scrolling
        start_col = max(0, first_showing_col)
        end_col = min(n_cols, last_showing_col + 10)  # Buffer for columns

        for row in range(start_row, end_row):
            row_has_visible = False
            for col in range(start_col, end_col):
                try:
                    cell = table_iface.getAccessibleAt(row, col)
                    if cell is None:
                        continue

                    if not cell.getState().contains(pyatspi.STATE_SHOWING):
                        continue

                    row_has_visible = True
                    bounds = _get_bounds_minimal(cell)
                    if bounds:
                        x, y, w, h = bounds
                        cell_name = (cell.name or "").strip()
                        cell_text = _get_text_minimal(cell)
                        cell_role = (cell.getRoleName() or "").strip().lower()

                        cell_elem = {
                            "role": cell_role,
                            "name": cell_name,
                            "bounds": {"x": x, "y": y, "w": w, "h": h},
                            "center": {"x": x + w // 2, "y": y + h // 2},
                        }
                        if cell_text:
                            cell_elem["text"] = cell_text
                        # Include app/window info inherited from parent
                        if inherited_app:
                            cell_elem["app"] = inherited_app
                        if inherited_window:
                            cell_elem["window"] = inherited_window
                        children.append(cell_elem)
                except Exception:
                    pass

            # If we've found visible rows and this row has none, we might be past the visible area
            if last_showing_row is not None and row > last_showing_row + 5 and not row_has_visible:
                break

        return children

    # Roles that are pure containers with no semantic value - skip if unnamed
    CONTAINER_ROLES = {'panel', 'filler', 'section', 'redundant object', 'unknown', 'scroll pane'}
    # Roles that should always be skipped (decorative/structural only)
    SKIP_ROLES = {'separator'}
    # Generic actions that don't indicate meaningful interactivity
    GENERIC_ACTIONS = {'doDefault', 'showContextMenu', 'click', 'press', 'release'}

    def has_meaningful_actions(actions: list) -> bool:
        """Check if actions list contains non-generic actions."""
        if not actions:
            return False
        return any(a not in GENERIC_ACTIONS for a in actions)

    def build_tree(node: Accessible, depth: int = 0, in_calc: bool = False, inherited_app: str = None, inherited_window: str = None) -> dict | None:
        """Recursively build a nested tree structure from an AT-SPI node.

        Args:
            node: The AT-SPI accessible node
            depth: Current depth in the tree
            in_calc: Whether we're inside a LibreOffice Calc document
            inherited_app: App name inherited from parent (for broken parent chains)
            inherited_window: Window title inherited from parent
        """
        if depth > max_depth:
            return None

        role = (node.getRoleName() or "").strip().lower()
        is_showing = _is_showing_minimal(node)

        # Skip decorative/structural roles entirely
        if role in SKIP_ROLES:
            return None

        # Special handling for menu items: only include if parent menu is expanded
        # GTK caches menu item bounds even when menus are closed, causing stale data
        if role in ("menu item", "check menu item", "radio menu item"):
            # Check if parent menu is expanded (selected in its parent's selection)
            try:
                parent = node.parent
                if parent:
                    parent_role = (parent.getRoleName() or "").strip().lower()
                    if parent_role == "menu":
                        # Check the grandparent to determine the context
                        grandparent = parent.parent
                        grandparent_role = (grandparent.getRoleName() or "").strip().lower() if grandparent else ""

                        if grandparent_role == "combo box":
                            # Combo box dropdown - include if the menu has STATE_SHOWING
                            parent_state = parent.getState()
                            if not parent_state.contains(pyatspi.STATE_SHOWING):
                                return None
                        elif grandparent_role in ("menu bar", "menu"):
                            # Menu bar or submenu - use selection-based check
                            if not _is_menu_expanded(parent):
                                return None
            except Exception:
                pass

        # For submenus (menu inside menu), only include if parent menu is expanded
        if role == "menu":
            try:
                parent = node.parent
                if parent:
                    parent_role = (parent.getRoleName() or "").strip().lower()
                    if parent_role == "menu":
                        # This is a submenu - only include if parent is expanded
                        if not _is_menu_expanded(parent):
                            return None
                    elif parent_role == "combo box":
                        # Combo box dropdown menu - include if it has STATE_SHOWING
                        state = node.getState()
                        if not state.contains(pyatspi.STATE_SHOWING):
                            return None
            except Exception:
                pass

        # Track app name and window title for passing down to children
        current_app = inherited_app
        current_window = inherited_window

        # If this is an application node, capture its name
        if role == "application":
            current_app = (node.name or "").strip()

        # If this is a frame/window, capture its title
        if role in ("frame", "dialog", "window") and node.name:
            current_window = (node.name or "").strip()

        # Detect LibreOffice Calc document
        if role == "document spreadsheet":
            in_calc = True

        # Check if this is a top-level window that should be filtered due to occlusion
        # Only apply to actual top-level windows (depth == 1, direct children of application)
        # Not to internal frames used for layout (like Qt frames in VLC)
        if filter_occluded and role in ("frame", "dialog", "window", "alert", "file chooser") and depth == 1:
            if node not in visible_windows:
                # Window is occluded - check if it's an overlay app (gnome-shell, gjs)
                # Overlay apps contain dock/panel UI and should always be included
                app_name_check = (current_app or "").lower()

                if app_name_check in OVERLAY_APPS:
                    # Overlay app window - include it (dock/panel UI)
                    pass
                else:
                    # Regular app window that's occluded - skip it
                    return None

        # First, try to build children - this allows us to include parent nodes
        # that aren't "showing" themselves but have visible children
        children = []

        # Warmup: Access node attributes to trigger AT-SPI cache population
        # This is needed for some apps like Chrome that use lazy initialization
        try:
            node.getState()
            node.get_attributes()
            node.queryComponent()
        except Exception:
            pass

        # Special handling for LibreOffice Calc tables - use optimized traversal
        if in_calc and role == "table":
            children = build_calc_table_children(node, current_app, current_window)
        else:
            try:
                # Use Python iterator instead of getChildAtIndex
                for child in node:
                    if child:
                        child_tree = build_tree(child, depth + 1, in_calc, current_app, current_window)
                        if child_tree:
                            children.append(child_tree)
            except Exception:
                pass

        # Now decide whether to include this node
        bounds = _get_bounds_minimal(node)
        name = (node.name or "").strip()
        # Use description as fallback for name (e.g., VLC media buttons have descriptions but no names)
        description = ""
        if not name:
            try:
                description = (node.description or "").strip()
                # Take first line of description if multiline
                if description and '\n' in description:
                    description = description.split('\n')[0].strip()
            except Exception:
                pass
        text = _get_text_minimal(node)

        # If still no name/description, try parent's description (e.g., LibreOffice sidebar buttons)
        # BUT: Don't use parent description for content roles (paragraph, section, etc.)
        # because for documents, we want to show the actual text content, not parent's description
        content_roles = {'paragraph', 'section', 'heading', 'block quote', 'article', 'document text', 'document frame'}
        if not name and not description and role not in content_roles:
            try:
                parent = node.parent
                if parent:
                    parent_desc = (parent.description or "").strip()
                    if parent_desc:
                        description = parent_desc
            except Exception:
                pass
        actions = _get_actions_minimal(node)

        # Get text selection for content roles
        selection = None
        if role in content_roles:
            selection = _get_text_selection(node)

        # Check if this is a "useless" container node:
        # - It's a container role (panel, filler, section)
        # - It has no name, no text, no description, and only generic actions
        # - If it has exactly 1 child, just return that child (flatten)
        # - If it has 0 children, skip it
        # - If multiple children, keep as container
        is_container = role in CONTAINER_ROLES
        has_content = bool(name or description or text or has_meaningful_actions(actions))

        if is_container and not has_content:
            if len(children) == 0:
                return None
            elif len(children) == 1:
                # Flatten: return the single child directly
                return children[0]
            # Multiple children - filter out empty containers from children
            # Keep children that have: children, name, text, actions, OR valid bounds for interactive roles
            INTERACTIVE_CHILD_ROLES = {
                'scroll bar', 'slider', 'spin button', 'toggle button', 'push button',
                'button', 'check button', 'radio button', 'check box', 'combo box'
            }
            def is_meaningful_child(c):
                if c.get('children') or c.get('name') or c.get('text') or c.get('actions'):
                    return True
                # Also keep children with valid bounds that are interactive
                c_role = c.get('role', '')
                c_bounds = c.get('bounds', {})
                if c_role in INTERACTIVE_CHILD_ROLES and c_bounds.get('w', 0) > 0:
                    return True
                return False
            non_empty_children = [c for c in children if is_meaningful_child(c)]
            if len(non_empty_children) == 0:
                return None
            elif len(non_empty_children) == 1:
                return non_empty_children[0]
            children = non_empty_children

        # If node is not showing and has no visible children, skip it
        # Exception: application nodes are containers and should be included if they have children
        # Exception: menu items in an expanded menu should be included
        is_menu_item_in_expanded = False
        if role in ("menu item", "check menu item", "radio menu item", "tearoff menu item"):
            try:
                parent = node.parent
                if parent and (parent.getRoleName() or "").strip().lower() == "menu":
                    is_menu_item_in_expanded = _is_menu_expanded(parent)
            except Exception:
                pass
        # Also check for submenus in expanded menus
        if role == "menu":
            try:
                parent = node.parent
                if parent and (parent.getRoleName() or "").strip().lower() == "menu":
                    is_menu_item_in_expanded = _is_menu_expanded(parent)
            except Exception:
                pass

        if not is_showing and role != "application" and not children and not is_menu_item_in_expanded:
            return None

        if bounds is None:
            # No bounds - only include if we have children
            if children:
                # Skip unnamed containers with no bounds
                if is_container and not has_content:
                    # Just return children wrapped minimally
                    if len(children) == 1:
                        return children[0]
                return {
                    "role": role,
                    "name": name,
                    "children": children
                }
            return None

        # Node has bounds - build full element
        x, y, w, h = bounds
        # Use inherited app/window names (passed down during traversal) as they're more reliable
        # than walking up parent chain (which is broken for some apps like LibreOffice)
        app_name = inherited_app
        window_title = inherited_window
        # Fall back to walking up parent chain if no inherited values
        if not app_name or not window_title:
            walked_app, walked_window = _get_window_context_minimal(node)
            if not app_name:
                app_name = walked_app
            if not window_title:
                window_title = walked_window

        # Use name, or description as fallback
        display_name = name or description

        # Build element data
        elem = {
            "role": role,
            "name": display_name,
            "bounds": {"x": x, "y": y, "w": w, "h": h},
            "center": {"x": x + w // 2, "y": y + h // 2},
        }

        # Only include optional fields if they have values
        if text:
            elem["text"] = text
        # Include description separately if we have both name and description
        if name and description and description != name:
            elem["description"] = description
        # Filter out generic actions to reduce noise
        meaningful_actions = [a for a in actions if a not in GENERIC_ACTIONS]
        if meaningful_actions:
            elem["actions"] = meaningful_actions
        if app_name:
            elem["app"] = app_name
        if window_title:
            elem["window"] = window_title

        # Mark active/focused/disabled state
        try:
            state = node.getState()
            # For top-level windows, mark active/focused
            if role in ("frame", "dialog", "window"):
                if state.contains(pyatspi.STATE_ACTIVE):
                    elem["active"] = True
                if state.contains(pyatspi.STATE_FOCUSED):
                    elem["focused"] = True
            # For interactive elements, mark if disabled
            # An element is disabled if it's NOT enabled or NOT sensitive
            if not state.contains(pyatspi.STATE_ENABLED) or not state.contains(pyatspi.STATE_SENSITIVE):
                # Only mark as disabled for interactive roles (buttons, entries, etc.)
                interactive_roles = {'push button', 'button', 'toggle button', 'check button',
                                    'radio button', 'check box', 'entry', 'text', 'combo box',
                                    'spin button', 'slider', 'link', 'menu item', 'list item'}
                if role in interactive_roles:
                    elem["disabled"] = True

            # For checkable elements (checkboxes, radio buttons, toggle buttons), mark checked state
            checkable_roles = {'check box', 'check button', 'radio button', 'toggle button',
                              'check menu item', 'radio menu item'}
            if role in checkable_roles:
                if state.contains(pyatspi.STATE_CHECKED):
                    elem["checked"] = True
                else:
                    elem["checked"] = False

            # For selectable items (list items, tree items), mark selected state
            selectable_roles = {'list item', 'tree item', 'table cell', 'table row', 'menu item'}
            if role in selectable_roles:
                if state.contains(pyatspi.STATE_SELECTED):
                    elem["selected"] = True

            # For text/entry fields, mark if editable and get caret position
            # Only show caret for actually editable elements, not labels
            editable_roles = ('text', 'entry', 'combo box', 'spin button', 'password text', 'paragraph')
            if role in editable_roles and state.contains(pyatspi.STATE_EDITABLE):
                elem["editable"] = True

                # For editable elements, get caret position if focused
                try:
                    text_iface = node.queryText()
                    if text_iface:
                        caret_offset = text_iface.caretOffset
                        if caret_offset >= 0:
                            elem["focused"] = True
                            elem["caret_offset"] = caret_offset
                            # Get caret position in screen coordinates
                            try:
                                rect = text_iface.getCharacterExtents(caret_offset, 0)  # 0 = screen coords
                                if rect and len(rect) >= 2:
                                    elem["caret"] = {"x": rect[0], "y": rect[1]}
                                    if len(rect) >= 4:
                                        elem["caret"]["w"] = rect[2]
                                        elem["caret"]["h"] = rect[3]
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

        # For scroll bars and sliders, get the current value
        if role in ('scroll bar', 'slider', 'spin button'):
            try:
                vi = node.queryValue()
                if vi:
                    elem["value"] = round(vi.currentValue, 2)
                    elem["min_value"] = round(vi.minimumValue, 2)
                    elem["max_value"] = round(vi.maximumValue, 2)
            except Exception:
                pass

        # Add text selection info for content roles
        if selection:
            elem["selection"] = selection

        # Add children (already built above)
        if children:
            elem["children"] = children

        return elem

    def flatten_tree(node):
        """Post-process to flatten chains of single-child containers."""
        if not isinstance(node, dict):
            return node

        children = node.get('children', [])
        # Recursively flatten children first
        children = [flatten_tree(c) for c in children if c is not None]
        children = [c for c in children if c is not None]

        role = node.get('role', '')
        name = node.get('name', '')
        text = node.get('text', '')
        actions = node.get('actions', [])

        # Check if this is a pure container with no semantic content
        has_content = bool(name or text or actions)

        # For container roles (panel, section, filler, etc.), flatten aggressively
        if role in CONTAINER_ROLES and not has_content:
            # Filter out children that are empty containers
            non_empty = []
            for c in children:
                c_role = c.get('role', '')
                c_has_content = bool(c.get('name') or c.get('text') or c.get('actions'))
                c_has_children = bool(c.get('children'))
                if c_has_content or c_has_children or c_role not in CONTAINER_ROLES:
                    non_empty.append(c)

            if len(non_empty) == 0:
                return None
            elif len(non_empty) == 1:
                # Flatten: return the single meaningful child
                return non_empty[0]
            children = non_empty

        # Update children
        if children:
            node['children'] = children
        elif 'children' in node:
            del node['children']

        return node

    # Build tree for each application under the desktop
    apps = []
    try:
        for app_node in desktop:
            # Skip background service apps (no visible UI)
            app_name = (app_node.name or "").strip().lower()
            if app_name in BACKGROUND_APPS:
                continue

            app_tree = build_tree(app_node)
            if app_tree:
                # Post-process to flatten chains
                app_tree = flatten_tree(app_tree)
                # Only include apps that have visible children
                if app_tree and app_tree.get("children"):
                    apps.append(app_tree)
    except Exception as e:
        logger.error(f"Error during AT-SPI traversal: {e}")

    # If flat mode is requested, extract actionable elements and important content
    if flat_mode:
        # Roles that can be interacted with
        ACTIONABLE_ROLES = {
            'push button', 'button', 'toggle button', 'check button', 'radio button',
            'link', 'menu item', 'check menu item', 'radio menu item',
            'entry', 'text', 'password text', 'spin button', 'combo box',
            'slider', 'scroll bar', 'list item', 'tree item', 'tab', 'page tab',
            'menu', 'menu bar', 'tool bar', 'table cell', 'icon',
            'frame', 'dialog', 'window'  # Include top-level windows so apps are visible even without children
        }
        # Roles that contain important content/context
        CONTENT_ROLES = {
            'static', 'label', 'heading', 'paragraph', 'block quote',
            'article', 'caption', 'description', 'alert', 'terminal'
        }
        # Note: 'section' removed - Chrome sections duplicate content already in static/heading elements

        # Roles that are interactive even without names (media controls, etc.)
        INTERACTIVE_ROLES = {
            'push button', 'button', 'toggle button', 'check button', 'radio button',
            'check box', 'slider', 'spin button', 'combo box', 'entry', 'text',
            'scroll bar'
        }

        # Container roles that should include their children (lists, tables, trees, menus, etc.)
        CONTAINER_WITH_ITEMS = {'list', 'list box', 'tree', 'tree table', 'table', 'layered pane', 'document text', 'document frame', 'document', 'scroll pane', 'document presentation', 'menu', 'menu bar', 'combo box', 'dialog', 'alert', 'file chooser'}
        ITEM_ROLES = {'list item', 'tree item', 'table cell', 'table row', 'canvas', 'icon', 'paragraph', 'shape', 'panel', 'menu item', 'check menu item', 'radio menu item'}
        # Menu roles that can be both containers AND items (submenus)
        MENU_ROLES = {'menu', 'menu item', 'check menu item', 'radio menu item'}
        # Note: 'section' removed from ITEM_ROLES - Chrome uses section for layout, not content
        # Note: 'panel' added for LibreOffice Impress presentation placeholders (PresentationTitle, PresentationSubtitle)
        # Note: 'menu', 'menu bar' added as containers, 'menu item' variants added as items

        def is_valid_bounds(b):
            """Check if bounds are valid (on-screen, positive coordinates)."""
            if not b:
                return False
            x, y, w, h = b.get('x', 0), b.get('y', 0), b.get('w', 0), b.get('h', 0)
            # Filter out invalid bounds (negative coords, zero size, or obviously offscreen)
            if x < -1000 or y < -1000 or w <= 0 or h <= 0:
                return False
            # Filter out bounds that are way outside screen (normalized > 2.0 would be way off)
            if dw > 0 and dh > 0:
                if x / dw > 2.0 or y / dh > 2.0:
                    return False
            return True

        # Pre-collect cell editing panels (panels named "Cell X#" with editable paragraphs)
        # These are overlay panels that appear when editing a spreadsheet cell
        cell_editing_panels = {}  # Maps cell name (e.g., "A1") to editing content

        def collect_cell_editing_panels(node):
            """Recursively collect cell editing panels from the tree."""
            role = node.get('role', '')
            name = node.get('name', '')

            # Check if this is a cell editing panel (pattern: "Cell A1", "Cell B2", etc.)
            if role == 'panel' and name and name.startswith('Cell '):
                cell_name = name[5:]  # Extract "A1" from "Cell A1"
                # Look for editable paragraph inside
                for child in node.get('children', []):
                    if child.get('role') == 'paragraph':
                        para_text = child.get('text', '')
                        para_bounds = child.get('bounds')
                        if para_text or child.get('editable'):
                            cell_editing_panels[cell_name] = {
                                'text': para_text,
                                'bounds': para_bounds,
                                'editable': child.get('editable', False),
                                'name': child.get('name', ''),
                                'caret_offset': child.get('caret_offset'),  # Include caret position
                                'focused': child.get('focused', False),
                            }
                            break

            for child in node.get('children', []):
                collect_cell_editing_panels(child)

        # Collect editing panels from all apps
        for app in apps:
            collect_cell_editing_panels(app)

        def extract_actionable(node, results=None, parent_is_container=False):
            if results is None:
                results = []

            role = node.get('role', '')
            name = node.get('name', '')
            text = node.get('text', '')
            bounds = node.get('bounds')
            children = node.get('children', [])


            # Handle container elements (lists, tables, trees) - extract with their items nested
            if role in CONTAINER_WITH_ITEMS and bounds:
                # Collect items from this container
                items = []
                # Also collect sibling elements like scrollbars that should be nested
                sibling_elements = []
                # Track nested containers (e.g., table inside scroll pane)
                nested_containers = []

                for child in children:
                    child_role = child.get('role', '')
                    child_bounds = child.get('bounds')
                    child_name = child.get('name', '')

                    # Include scrollbars as sibling elements within the container
                    if child_role in ('scroll bar', 'slider') and is_valid_bounds(child_bounds):
                        scrollbar = {
                            'role': child_role,
                            'name': child_name or f'[{child_role}]',
                            'bounds': child_bounds,
                            'center': child.get('center'),
                        }
                        if child.get('value') is not None:
                            scrollbar['value'] = child['value']
                        sibling_elements.append(scrollbar)
                    # Handle menus inside menu bar, other menus (submenus), or combo boxes (dropdowns)
                    # Note: combo box dropdown menus often have empty names, so we don't require child_name for combo boxes
                    elif child_role == 'menu' and is_valid_bounds(child_bounds) and (child_name or role == 'combo box'):
                        # Extract menu items from this menu
                        menu_items = []
                        for gc in child.get('children', []):
                            gc_role = gc.get('role', '')
                            gc_name = gc.get('name', '')
                            gc_bounds = gc.get('bounds')

                            if gc_role in MENU_ROLES and is_valid_bounds(gc_bounds) and gc_name:
                                menu_item = {
                                    'role': gc_role,
                                    'name': gc_name,
                                    'bounds': gc_bounds,
                                    'center': gc.get('center'),
                                }
                                if gc.get('disabled'):
                                    menu_item['disabled'] = True
                                if 'checked' in gc:
                                    menu_item['checked'] = gc['checked']
                                # If this is a submenu, recursively extract its items
                                if gc_role == 'menu':
                                    submenu_items = []
                                    for ggc in gc.get('children', []):
                                        ggc_role = ggc.get('role', '')
                                        ggc_name = ggc.get('name', '')
                                        ggc_bounds = ggc.get('bounds')
                                        if ggc_role in MENU_ROLES and is_valid_bounds(ggc_bounds) and ggc_name:
                                            sub_item = {
                                                'role': ggc_role,
                                                'name': ggc_name,
                                                'bounds': ggc_bounds,
                                                'center': ggc.get('center'),
                                            }
                                            if ggc.get('disabled'):
                                                sub_item['disabled'] = True
                                            submenu_items.append(sub_item)
                                    if submenu_items:
                                        menu_item['items'] = submenu_items
                                menu_items.append(menu_item)

                        menu_elem = {
                            'role': child_role,
                            'name': child_name,
                            'bounds': child_bounds,
                            'center': child.get('center'),
                        }
                        if menu_items:
                            menu_elem['items'] = menu_items
                        items.append(menu_elem)
                    elif child_role in ITEM_ROLES:
                        child_name = child.get('name', '')
                        child_text = child.get('text', '')

                        # For panels (like presentation placeholders), look for text in child paragraphs
                        if child_role == 'panel' and not child_text:
                            debug_log(f"[PANEL] '{child_name}' has {len(child.get('children', []))} children")
                            for gc in child.get('children', []):
                                gc_role = gc.get('role', '')
                                gc_text = gc.get('text', '')
                                debug_log(f"  gc: [{gc_role}] text={gc_text!r}")
                                if gc_role == 'paragraph':
                                    if gc_text:
                                        child_text = gc_text
                                        debug_log(f"  -> Extracted: {child_text!r}")
                                        break

                        # For paragraph/section, use text as the display content
                        # Include if there's any text content (not just name) and valid bounds
                        if is_valid_bounds(child_bounds) and (child_name or child_text):
                            # For document content roles, prefer text over name
                            display_name = child_name
                            if child_role in ('paragraph', 'section') and child_text:
                                display_name = child_text[:100]  # Truncate very long text
                            # For panels with text content (presentation placeholders), show the text
                            if child_role == 'panel' and child_text:
                                display_name = child_name or child_text[:100]
                            item = {
                                'role': child_role,
                                'name': display_name,
                                'bounds': child_bounds,
                                'center': child.get('center'),
                            }
                            # Include full text if different from display name
                            if child_text and child_text != display_name:
                                item['text'] = child_text
                                debug_log(f"[TEXT] Added to '{display_name}': {child_text!r}")
                            if child.get('disabled'):
                                item['disabled'] = True
                            if 'checked' in child:
                                item['checked'] = child['checked']
                            # Check for selected state
                            if child.get('selected'):
                                item['selected'] = True
                            # Include text selection info
                            if child.get('selection'):
                                item['selection'] = child['selection']
                            # Include editable state and caret for paragraphs/text
                            if child.get('editable'):
                                item['editable'] = True
                            if child.get('focused'):
                                item['focused'] = True
                            if child.get('caret_offset') is not None:
                                item['caret_offset'] = child['caret_offset']
                            if child.get('caret'):
                                item['caret'] = child['caret']

                            # For table cells, check if there's an associated editing panel
                            if child_role == 'table cell' and child_name in cell_editing_panels:
                                editing = cell_editing_panels[child_name]
                                if editing.get('text') is not None:  # Allow empty string
                                    edit_text = editing['text']
                                    caret_offset = editing.get('caret_offset')
                                    # Insert caret marker if we have a valid caret position
                                    if caret_offset is not None and caret_offset >= 0:
                                        offset = min(caret_offset, len(edit_text))
                                        edit_text = edit_text[:offset] + '<caret/>' + edit_text[offset:]
                                        item['focused'] = True
                                    item['editing'] = edit_text
                                    item['editable'] = True

                            items.append(item)
                    elif child_role in CONTAINER_WITH_ITEMS and is_valid_bounds(child_bounds):
                        # Nested container (e.g., table inside scroll pane, or submenu inside menu)
                        # Extract its items and include them
                        nested_items = []
                        for grandchild in child.get('children', []):
                            gc_role = grandchild.get('role', '')
                            gc_name = grandchild.get('name', '')
                            gc_text = grandchild.get('text', '')
                            gc_bounds = grandchild.get('bounds')

                            # Handle nested submenus (menu inside menu)
                            if gc_role in ('menu',) and is_valid_bounds(gc_bounds) and gc_name:
                                # This is a submenu - recursively extract its items
                                submenu_items = []
                                for ggc in grandchild.get('children', []):
                                    ggc_role = ggc.get('role', '')
                                    ggc_name = ggc.get('name', '')
                                    ggc_bounds = ggc.get('bounds')
                                    if ggc_role in ITEM_ROLES and is_valid_bounds(ggc_bounds) and ggc_name:
                                        sub_item = {
                                            'role': ggc_role,
                                            'name': ggc_name,
                                            'bounds': ggc_bounds,
                                            'center': ggc.get('center'),
                                        }
                                        if ggc.get('disabled'):
                                            sub_item['disabled'] = True
                                        if 'checked' in ggc:
                                            sub_item['checked'] = ggc['checked']
                                        submenu_items.append(sub_item)
                                submenu = {
                                    'role': gc_role,
                                    'name': gc_name,
                                    'bounds': gc_bounds,
                                    'center': grandchild.get('center'),
                                }
                                if submenu_items:
                                    submenu['items'] = submenu_items
                                nested_items.append(submenu)
                            elif gc_role in ITEM_ROLES and is_valid_bounds(gc_bounds):
                                # For panels (like presentation placeholders), look for text in child paragraphs
                                # Also extract caret info from the focused paragraph
                                gc_caret = None
                                gc_caret_offset = None
                                gc_focused = False
                                gc_editable = False

                                # For paragraphs directly in containers (like document text), extract caret info
                                if gc_role == 'paragraph':
                                    if grandchild.get('editable'):
                                        gc_editable = True
                                    if grandchild.get('focused') and grandchild.get('caret_offset') is not None:
                                        gc_focused = True
                                    if grandchild.get('caret'):
                                        gc_caret = grandchild['caret']
                                    if grandchild.get('caret_offset') is not None:
                                        gc_caret_offset = grandchild['caret_offset']

                                if gc_role == 'panel' and not gc_text:
                                    debug_log(f"[NESTED PANEL] '{gc_name}' has {len(grandchild.get('children', []))} children")
                                    for ggc in grandchild.get('children', []):
                                        ggc_role = ggc.get('role', '')
                                        ggc_text = ggc.get('text', '')
                                        debug_log(f"  ggc: [{ggc_role}] text={ggc_text!r}")
                                        if ggc_role == 'paragraph' and ggc_text:
                                            gc_text = ggc_text
                                            # Copy caret info from the paragraph
                                            # Only mark as focused if there's a valid caret position
                                            if ggc.get('editable'):
                                                gc_editable = True
                                            if ggc.get('caret'):
                                                gc_caret = ggc['caret']
                                                gc_focused = True  # Only focused if has caret
                                            if ggc.get('caret_offset') is not None:
                                                gc_caret_offset = ggc['caret_offset']
                                            debug_log(f"  -> Extracted: {gc_text!r} focused={gc_focused} caret={gc_caret}")
                                            break

                                # Only include if has name or text
                                if gc_name or gc_text:
                                    display_name = gc_name
                                    if gc_role in ('paragraph', 'section') and gc_text:
                                        display_name = gc_text[:100]
                                    # For panels with text content, show the text
                                    if gc_role == 'panel' and gc_text:
                                        display_name = gc_name or gc_text[:100]
                                    item = {
                                        'role': gc_role,
                                        'name': display_name,
                                        'bounds': gc_bounds,
                                        'center': grandchild.get('center'),
                                    }
                                    if gc_text and gc_text != display_name:
                                        item['text'] = gc_text
                                        debug_log(f"[NESTED TEXT] Added to '{display_name}': {gc_text!r}")
                                    if grandchild.get('disabled'):
                                        item['disabled'] = True
                                    if 'checked' in grandchild:
                                        item['checked'] = grandchild['checked']
                                    if grandchild.get('selected'):
                                        item['selected'] = True
                                    if grandchild.get('selection'):
                                        item['selection'] = grandchild['selection']
                                    # Add caret info from child paragraph
                                    if gc_focused:
                                        item['focused'] = True
                                    if gc_editable:
                                        item['editable'] = True
                                    if gc_caret:
                                        item['caret'] = gc_caret
                                    if gc_caret_offset is not None:
                                        item['caret_offset'] = gc_caret_offset
                                    nested_items.append(item)
                        if nested_items:
                            # Use the nested container's role and bounds, with extracted items
                            nested_container = {
                                'role': child_role,
                                'name': child.get('name', ''),
                                'bounds': child_bounds,
                                'center': child.get('center'),
                                'items': nested_items,
                            }
                            nested_containers.append(nested_container)
                    # For dialogs, recursively collect ALL actionable elements from descendants
                    elif role in ('dialog', 'alert', 'file chooser'):
                        # Recursively collect actionable elements from this child
                        def collect_dialog_items(node, collected, in_combo=False):
                            n_role = node.get('role', '')
                            n_name = node.get('name', '')
                            n_text = node.get('text', '')
                            n_bounds = node.get('bounds')

                            # Skip menu items inside combo boxes (they're already nested)
                            if in_combo and n_role in ('menu', 'menu item'):
                                return

                            if n_role in ACTIONABLE_ROLES and is_valid_bounds(n_bounds) and (n_name or n_text or n_role in INTERACTIVE_ROLES):
                                display_name = n_name or n_text or f'[{n_role}]'
                                item = {
                                    'role': n_role,
                                    'name': display_name,
                                    'bounds': n_bounds,
                                    'center': node.get('center'),
                                }
                                if n_text and n_text != display_name:
                                    item['text'] = n_text
                                if node.get('disabled'):
                                    item['disabled'] = True
                                if 'checked' in node:
                                    item['checked'] = node['checked']
                                if node.get('editable'):
                                    item['editable'] = True
                                # For combo boxes with dropdown, include their children
                                if n_role == 'combo box' and node.get('children'):
                                    combo_items = []
                                    for gc in node.get('children', []):
                                        if gc.get('role') == 'menu':
                                            for mi in gc.get('children', []):
                                                mi_name = mi.get('name', '')
                                                mi_bounds = mi.get('bounds')
                                                if mi_name and is_valid_bounds(mi_bounds):
                                                    combo_items.append({
                                                        'role': mi.get('role', 'menu item'),
                                                        'name': mi_name,
                                                        'bounds': mi_bounds,
                                                        'center': mi.get('center'),
                                                    })
                                    if combo_items:
                                        item['items'] = combo_items
                                collected.append(item)

                                # If this is a combo box, don't recurse into its children
                                # (menu items are already extracted above)
                                if n_role == 'combo box':
                                    return

                            # Recurse into children
                            for gc in node.get('children', []):
                                collect_dialog_items(gc, collected, in_combo=(n_role == 'combo box'))

                        collect_dialog_items(child, items)
                    else:
                        # Recurse into non-item children
                        extract_actionable(child, results, parent_is_container=True)

                # For scroll pane, merge nested containers with scrollbars
                if role == 'scroll pane' and nested_containers and sibling_elements:
                    # Add scrollbars to each nested container
                    for nc in nested_containers:
                        nc['controls'] = sibling_elements
                        if node.get('app'):
                            nc['app'] = node.get('app')
                        results.append(nc)
                    return results

                # Only add the container if it has items
                if items or sibling_elements or nested_containers:
                    container_elem = {
                        'role': role,
                        'name': name,
                        'bounds': bounds,
                        'center': node.get('center'),
                    }
                    if items:
                        container_elem['items'] = items
                    # Add scrollbars/sliders as nested elements
                    if sibling_elements:
                        container_elem['controls'] = sibling_elements
                    # Add nested containers as children
                    if nested_containers:
                        container_elem['nested'] = nested_containers
                    if node.get('app'):
                        container_elem['app'] = node.get('app')
                    results.append(container_elem)
                return results

            # Include if it's an actionable role with name or text
            if role in ACTIONABLE_ROLES and (name or text) and is_valid_bounds(bounds):
                elem = {
                    'role': role,
                    'name': name,
                    'bounds': bounds,
                    'center': node.get('center'),
                }
                if text and text != name:
                    elem['text'] = text
                if node.get('app'):
                    elem['app'] = node.get('app')
                if node.get('disabled'):
                    elem['disabled'] = True
                if 'checked' in node:
                    elem['checked'] = node['checked']
                results.append(elem)
            # Include interactive controls even without names (e.g., VLC media buttons)
            # but only if they have valid bounds
            # Note: name may already include description from build_tree
            elif role in INTERACTIVE_ROLES and is_valid_bounds(bounds):
                # For text/entry/combo box fields, use text content as display name if no name
                # This makes dropdown values like "12 pt" show as the name
                if role in ('text', 'entry', 'combo box') and text and not name:
                    display_name = text[:50]  # Truncate long text
                else:
                    display_name = name if name else f'[{role}]'
                elem = {
                    'role': role,
                    'name': display_name,
                    'bounds': bounds,
                    'center': node.get('center'),
                }
                # Include text only if different from display name
                if text and text != display_name:
                    elem['text'] = text
                if node.get('description') and node.get('description') != name:
                    elem['description'] = node.get('description')
                if node.get('app'):
                    elem['app'] = node.get('app')
                if node.get('disabled'):
                    elem['disabled'] = True
                if 'checked' in node:
                    elem['checked'] = node['checked']
                # Mark editable fields
                if node.get('editable'):
                    elem['editable'] = True
                # Include focused state and caret position
                if node.get('focused'):
                    elem['focused'] = True
                if node.get('caret'):
                    elem['caret'] = node['caret']
                if node.get('caret_offset') is not None:
                    elem['caret_offset'] = node['caret_offset']
                # Include value for scroll bars/sliders
                if node.get('value') is not None:
                    elem['value'] = node['value']
                if node.get('min_value') is not None:
                    elem['min_value'] = node['min_value']
                if node.get('max_value') is not None:
                    elem['max_value'] = node['max_value']
                results.append(elem)
            # Include content roles if they have text content
            # Previously filtered by len(text) > 20 but this was too arbitrary
            # Now include all text content - the agent can decide what's relevant
            elif role in CONTENT_ROLES and text:
                elem = {
                    'role': role,
                    'name': name,
                    'text': text,
                    'bounds': bounds,
                    'center': node.get('center'),
                }
                if node.get('app'):
                    elem['app'] = node.get('app')
                if node.get('disabled'):
                    elem['disabled'] = True
                # Include focused state and caret position for editable content
                if node.get('focused'):
                    elem['focused'] = True
                if node.get('editable'):
                    elem['editable'] = True
                if node.get('caret'):
                    elem['caret'] = node['caret']
                if node.get('caret_offset') is not None:
                    elem['caret_offset'] = node['caret_offset']
                results.append(elem)

            for child in children:
                extract_actionable(child, results, parent_is_container=False)

            return results

        actionable = []
        for app in apps:
            extract_actionable(app, actionable)

        # Get actual screen dimensions from X11 display
        try:
            d = display.Display()
            screen_w = d.screen().width_in_pixels
            screen_h = d.screen().height_in_pixels
            if screen_w > 0 and screen_h > 0:
                dw = screen_w
                dh = screen_h
        except Exception:
            pass

        # Fallback to common screen sizes if still invalid
        if dw <= 0:
            dw = 1920
        if dh <= 0:
            dh = 1080

        # Group elements by app
        by_app = {}
        for elem in actionable:
            app_name = elem.pop('app', 'unknown')
            if app_name not in by_app:
                by_app[app_name] = []
            by_app[app_name].append(elem)

        # Check if XML format is requested
        output_format = request.args.get('format', 'xml').lower()

        if output_format == 'xml':
            # Build XML output
            def escape_xml(s):
                if s is None:
                    return ''
                return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

            xml_parts = [f'<desktop screen_w="{dw}" screen_h="{dh}">']
            for app_name, elements in by_app.items():
                xml_parts.append(f'  <app name="{escape_xml(app_name)}">')
                for elem in elements:
                    def format_element_xml(elem, indent='    '):
                        """Format a single element as XML, with nested items if present."""
                        role = escape_xml(elem.get('role', ''))
                        name = escape_xml(elem.get('name', ''))
                        text = escape_xml(elem.get('text', ''))
                        bounds = elem.get('bounds', {})
                        center = elem.get('center', {})
                        items = elem.get('items', [])

                        # Format bounds as box attribute [x,y,w,h] normalized to 0-1
                        if bounds and dw > 0 and dh > 0:
                            bx = bounds.get('x', 0) / dw
                            by = bounds.get('y', 0) / dh
                            bw = bounds.get('w', 0) / dw
                            bh = bounds.get('h', 0) / dh
                            box_attr = f'box="[{bx:.3f},{by:.3f},{bw:.3f},{bh:.3f}]"'
                        else:
                            box_attr = ''

                        # Format center as normalized coordinates
                        if center and dw > 0 and dh > 0:
                            cx = center.get('x', 0) / dw
                            cy = center.get('y', 0) / dh
                            center_attr = f'center="[{cx:.3f},{cy:.3f}]"'
                        else:
                            center_attr = ''

                        # Build element tag
                        attrs = []
                        if name:
                            attrs.append(f'name="{name}"')
                        if box_attr:
                            attrs.append(box_attr)
                        if center_attr:
                            attrs.append(center_attr)
                        # Add disabled attribute if element is disabled
                        if elem.get('disabled'):
                            attrs.append('disabled="true"')
                        # Add checked attribute for checkboxes/radio buttons
                        if 'checked' in elem:
                            attrs.append(f'checked="{str(elem["checked"]).lower()}"')
                        # Add selected attribute for list items
                        if elem.get('selected'):
                            attrs.append('selected="true"')
                        # Add selection info for text content
                        if elem.get('selection'):
                            sel = elem['selection']
                            sel_text = escape_xml(sel.get('text', ''))
                            attrs.append(f'selection="{sel_text}"')
                        # Add editable attribute for text fields
                        if elem.get('editable'):
                            attrs.append('editable="true"')
                        # Add focused attribute for focused elements
                        if elem.get('focused'):
                            attrs.append('focused="true"')
                        # Note: caret position is now shown inline in text as <caret/>
                        # Add editing content for cells being edited (with caret marker preserved)
                        if elem.get('editing'):
                            editing_text = escape_xml(elem['editing']).replace('&lt;caret/&gt;', '<caret/>')
                            attrs.append(f'editing="{editing_text}"')
                        # Add value for scroll bars/sliders
                        if elem.get('value') is not None:
                            attrs.append(f'value="{elem["value"]}"')

                        attr_str = ' '.join(attrs)

                        # Helper to insert <caret/> into text at the correct position
                        def insert_caret_marker(txt, caret_offset):
                            """Insert <caret/> marker at the specified character offset."""
                            if caret_offset is None or caret_offset < 0:
                                return txt
                            # Ensure offset is within bounds
                            offset = min(caret_offset, len(txt))
                            return txt[:offset] + '<caret/>' + txt[offset:]

                        # Get display text with caret marker if applicable
                        display_text = text
                        has_caret_in_name = False
                        caret_offset = elem.get('caret_offset')
                        if caret_offset is not None and elem.get('focused'):
                            # Insert caret marker into text
                            # For paragraphs in containers, text content may be in 'name' or 'text'
                            raw_text = elem.get('text', '') or name
                            if raw_text:
                                marked_text = insert_caret_marker(raw_text, caret_offset)
                                # Escape XML but preserve <caret/>
                                display_text = escape_xml(marked_text).replace('&lt;caret/&gt;', '<caret/>')
                                # If text was in 'name', we need to show it as content
                                if not elem.get('text') and name:
                                    has_caret_in_name = True

                        # If this element has nested items (list, table, tree) or controls (scrollbars)
                        controls = elem.get('controls', [])
                        if items or controls:
                            lines = [f'{indent}<{role} {attr_str}>']
                            for item in items:
                                lines.append(format_element_xml(item, indent + '  '))
                            # Add controls (scrollbars, sliders) after items
                            for ctrl in controls:
                                lines.append(format_element_xml(ctrl, indent + '  '))
                            lines.append(f'{indent}</{role}>')
                            return '\n'.join(lines)
                        elif display_text and (display_text != name or has_caret_in_name):
                            return f'{indent}<{role} {attr_str}>{display_text}</{role}>'
                        else:
                            return f'{indent}<{role} {attr_str} />'

                    xml_parts.append(format_element_xml(elem))

                xml_parts.append('  </app>')
            xml_parts.append('</desktop>')

            return '\n'.join(xml_parts), 200, {'Content-Type': 'application/xml'}

        # JSON format (default) - grouped by app
        payload = {
            "screen": {
                "x": dx,
                "y": dy,
                "width": dw,
                "height": dh,
            },
            "apps": by_app,
            "total_elements": len(actionable),
            "filter_occluded": filter_occluded,
            "occlusion_mode": occlusion_mode,
        }
        return jsonify(payload)

    payload = {
        "screen": {
            "x": dx,
            "y": dy,
            "width": dw,
            "height": dh,
        },
        "desktop": {
            "role": "desktop",
            "name": "Desktop",
            "children": apps
        },
        "filter_occluded": filter_occluded,
        "occlusion_mode": occlusion_mode,
        "total_windows": len(top_level_windows),
        "visible_windows": len(visible_windows),
    }

    return jsonify(payload)


@app.route('/screen_size', methods=['POST'])
def get_screen_size():
    if platform_name == "Linux":
        d = display.Display()
        screen_width = d.screen().width_in_pixels
        screen_height = d.screen().height_in_pixels
    elif platform_name == "Windows":
        user32 = ctypes.windll.user32
        screen_width: int = user32.GetSystemMetrics(0)
        screen_height: int = user32.GetSystemMetrics(1)
    return jsonify(
        {
            "width": screen_width,
            "height": screen_height
        }
    )


@app.route('/window_size', methods=['POST'])
def get_window_size():
    if 'app_class_name' in request.form:
        app_class_name = request.form['app_class_name']
    else:
        return jsonify({"error": "app_class_name is required"}), 400

    d = display.Display()
    root = d.screen().root
    window_ids = root.get_full_property(d.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType).value

    for window_id in window_ids:
        try:
            window = d.create_resource_object('window', window_id)
            wm_class = window.get_wm_class()

            if wm_class is None:
                continue

            if app_class_name.lower() in [name.lower() for name in wm_class]:
                geom = window.get_geometry()
                return jsonify(
                    {
                        "width": geom.width,
                        "height": geom.height
                    }
                )
        except Xlib.error.XError:  # Ignore windows that give an error
            continue
    return None


@app.route('/desktop_path', methods=['POST'])
def get_desktop_path():
    # Get the home directory in a platform-independent manner using pathlib
    home_directory = str(Path.home())

    # Determine the desktop path based on the operating system
    desktop_path = {
        "Windows": os.path.join(home_directory, "Desktop"),
        "Darwin": os.path.join(home_directory, "Desktop"),  # macOS
        "Linux": os.path.join(home_directory, "Desktop")
    }.get(platform.system(), None)

    # Check if the operating system is supported and the desktop path exists
    if desktop_path and os.path.exists(desktop_path):
        return jsonify(desktop_path=desktop_path)
    else:
        return jsonify(error="Unsupported operating system or desktop path not found"), 404


@app.route('/wallpaper', methods=['POST'])
def get_wallpaper():
    def get_wallpaper_windows():
        SPI_GETDESKWALLPAPER = 0x73
        MAX_PATH = 260
        buffer = ctypes.create_unicode_buffer(MAX_PATH)
        ctypes.windll.user32.SystemParametersInfoW(SPI_GETDESKWALLPAPER, MAX_PATH, buffer, 0)
        return buffer.value

    def get_wallpaper_macos():
        script = """
        tell application "System Events" to tell every desktop to get picture
        """
        process = subprocess.Popen(['osascript', '-e', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            app.logger.error("Error: %s", error.decode('utf-8'))
            return None
        return output.strip().decode('utf-8')

    def get_wallpaper_linux():
        try:
            output = subprocess.check_output(
                ["gsettings", "get", "org.gnome.desktop.background", "picture-uri"],
                stderr=subprocess.PIPE
            )
            return output.decode('utf-8').strip().replace('file://', '').replace("'", "")
        except subprocess.CalledProcessError as e:
            app.logger.error("Error: %s", e)
            return None

    os_name = platform.system()
    wallpaper_path = None
    if os_name == 'Windows':
        wallpaper_path = get_wallpaper_windows()
    elif os_name == 'Darwin':
        wallpaper_path = get_wallpaper_macos()
    elif os_name == 'Linux':
        wallpaper_path = get_wallpaper_linux()
    else:
        app.logger.error(f"Unsupported OS: {os_name}")
        abort(400, description="Unsupported OS")

    if wallpaper_path:
        try:
            # Ensure the filename is secure
            return send_file(wallpaper_path, mimetype='image/png')
        except Exception as e:
            app.logger.error(f"An error occurred while serving the wallpaper file: {e}")
            abort(500, description="Unable to serve the wallpaper file")
    else:
        abort(404, description="Wallpaper file not found")


@app.route('/list_directory', methods=['POST'])
def get_directory_tree():
    def _list_dir_contents(directory):
        """
        List the contents of a directory recursively, building a tree structure.

        :param directory: The path of the directory to inspect.
        :return: A nested dictionary with the contents of the directory.
        """
        tree = {'type': 'directory', 'name': os.path.basename(directory), 'children': []}
        try:
            # List all files and directories in the current directory
            for entry in os.listdir(directory):
                full_path = os.path.join(directory, entry)
                # If entry is a directory, recurse into it
                if os.path.isdir(full_path):
                    tree['children'].append(_list_dir_contents(full_path))
                else:
                    tree['children'].append({'type': 'file', 'name': entry})
        except OSError as e:
            # If the directory cannot be accessed, return the exception message
            tree = {'error': str(e)}
        return tree

    # Extract the 'path' parameter from the JSON request
    data = request.get_json()
    if 'path' not in data:
        return jsonify(error="Missing 'path' parameter"), 400

    start_path = data['path']
    # Ensure the provided path is a directory
    if not os.path.isdir(start_path):
        return jsonify(error="The provided path is not a directory"), 400

    # Generate the directory tree starting from the provided path
    directory_tree = _list_dir_contents(start_path)
    return jsonify(directory_tree=directory_tree)


@app.route('/file', methods=['POST'])
def get_file():
    # Retrieve filename from the POST request
    if 'file_path' in request.form:
        file_path = os.path.expandvars(os.path.expanduser(request.form['file_path']))
    else:
        return jsonify({"error": "file_path is required"}), 400

    try:
        # Check if the file exists and get its size
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        file_size = os.path.getsize(file_path)
        logger.info(f"Serving file: {file_path} ({file_size} bytes)")

        # Check if the file exists and send it to the user
        return send_file(file_path, as_attachment=True)
    except FileNotFoundError:
        # If the file is not found, return a 404 error
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error serving file {file_path}: {e}")
        return jsonify({"error": f"Failed to serve file: {str(e)}"}), 500


@app.route("/setup/upload", methods=["POST"])
def upload_file():
    # Retrieve filename from the POST request
    if 'file_path' in request.form and 'file_data' in request.files:
        file_path = os.path.expandvars(os.path.expanduser(request.form['file_path']))
        file = request.files["file_data"]

        try:
            # Ensure target directory exists
            target_dir = os.path.dirname(file_path)
            if target_dir:  # Only create directory if it's not empty
                os.makedirs(target_dir, exist_ok=True)

            # Save file and get size for verification
            file.save(file_path)
            uploaded_size = os.path.getsize(file_path)

            logger.info(f"File uploaded successfully: {file_path} ({uploaded_size} bytes)")
            return f"File Uploaded: {uploaded_size} bytes"

        except Exception as e:
            logger.error(f"Error uploading file to {file_path}: {e}")
            # Clean up partial file if it exists
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            return jsonify({"error": f"Failed to upload file: {str(e)}"}), 500
    else:
        return jsonify({"error": "file_path and file_data are required"}), 400


@app.route('/platform', methods=['GET'])
def get_platform():
    return platform.system()


@app.route('/cursor_position', methods=['GET'])
def get_cursor_position():
    pos = pyautogui.position()
    return jsonify(pos.x, pos.y)

@app.route("/setup/change_wallpaper", methods=['POST'])
def change_wallpaper():
    data = request.json
    path = data.get('path', None)

    if not path:
        return "Path not supplied!", 400

    path = Path(os.path.expandvars(os.path.expanduser(path)))

    if not path.exists():
        return f"File not found: {path}", 404

    try:
        user_platform = platform.system()
        if user_platform == "Windows":
            import ctypes
            ctypes.windll.user32.SystemParametersInfoW(20, 0, str(path), 3)
        elif user_platform == "Linux":
            import subprocess
            subprocess.run(["gsettings", "set", "org.gnome.desktop.background", "picture-uri", f"file://{path}"])
        elif user_platform == "Darwin":  # (Mac OS)
            import subprocess
            subprocess.run(
                ["osascript", "-e", f'tell application "Finder" to set desktop picture to POSIX file "{path}"'])
        return "Wallpaper changed successfully"
    except Exception as e:
        return f"Failed to change wallpaper. Error: {e}", 500


@app.route("/setup/download_file", methods=['POST'])
def download_file():
    data = request.json
    url = data.get('url', None)
    path = data.get('path', None)

    if not url or not path:
        return "Path or URL not supplied!", 400

    path = Path(os.path.expandvars(os.path.expanduser(path)))
    path.parent.mkdir(parents=True, exist_ok=True)

    max_retries = 3
    error: Optional[Exception] = None

    for i in range(max_retries):
        try:
            logger.info(f"Download attempt {i+1}/{max_retries} for {url}")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Get expected file size if available
            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                logger.info(f"Expected file size: {total_size / (1024*1024):.2f} MB")

            downloaded_size = 0
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0 and downloaded_size % (1024*1024) == 0:  # Log every MB
                            progress = (downloaded_size / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")

            # Verify download completeness
            actual_size = os.path.getsize(path)
            if total_size > 0 and actual_size != total_size:
                raise Exception(f"Download incomplete. Expected {total_size} bytes, got {actual_size} bytes")

            logger.info(f"File downloaded successfully: {path} ({actual_size} bytes)")
            return f"File downloaded successfully: {actual_size} bytes"

        except (requests.RequestException, Exception) as e:
            error = e
            logger.error(f"Failed to download {url}: {e}. Retrying... ({max_retries - i - 1} attempts left)")
            # Clean up partial download
            if path.exists():
                try:
                    path.unlink()
                except:
                    pass

    return f"Failed to download {url}. No retries left. Error: {error}", 500


@app.route("/setup/open_file", methods=['POST'])
def open_file():
    data = request.json
    path = data.get('path', None)

    if not path:
        return "Path not supplied!", 400

    path_obj = Path(os.path.expandvars(os.path.expanduser(path)))

    # Check if it's a file path that exists
    is_file_path = path_obj.exists()

    # If it's not a file path, treat it as an application name/command
    if not is_file_path:
        # Check if it's a valid command by trying to find it in PATH
        import shutil
        if not shutil.which(path):
            return f"Application/file not found: {path}", 404

    try:
        if is_file_path:
            # Handle file opening
            if platform.system() == "Windows":
                os.startfile(path_obj)
            else:
                open_cmd: str = "open" if platform.system() == "Darwin" else "xdg-open"
                subprocess.Popen([open_cmd, str(path_obj)])
            file_name = path_obj.name
            file_name_without_ext, _ = os.path.splitext(file_name)
        else:
            # Handle application launching
            if platform.system() == "Windows":
                subprocess.Popen([path])
            else:
                subprocess.Popen([path])
            file_name = path
            file_name_without_ext = path

        # Wait for the file/application to open

        start_time = time.time()
        window_found = False

        while time.time() - start_time < TIMEOUT:
            os_name = platform.system()
            if os_name in ['Windows', 'Darwin']:
                import pygetwindow as gw
                # Check for window title containing file name or file name without extension
                windows = gw.getWindowsWithTitle(file_name)
                if not windows:
                    windows = gw.getWindowsWithTitle(file_name_without_ext)

                if windows:
                    # To be more specific, we can try to activate it
                    windows[0].activate()
                    window_found = True
                    break
            elif os_name == 'Linux':
                try:
                    # Using wmctrl to list windows and check if any window title contains the filename
                    result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True, check=True)
                    window_list = result.stdout.strip().split('\n')
                    if not result.stdout.strip():
                        pass  # No windows, just continue waiting
                    else:
                        for window in window_list:
                            if file_name in window or file_name_without_ext in window:
                                # a window is found, now activate it
                                window_id = window.split()[0]
                                subprocess.run(['wmctrl', '-i', '-a', window_id], check=True)
                                window_found = True
                                break
                        if window_found:
                            break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # wmctrl might not be installed or the window manager isn't ready.
                    # We just log it once and let the main loop retry.
                    if 'wmctrl_failed_once' not in locals():
                        logger.warning("wmctrl command is not ready, will keep retrying...")
                        wmctrl_failed_once = True
                    pass  # Let the outer loop retry

            time.sleep(1)

        if window_found:
            return "File opened and window activated successfully"
        else:
            return f"Failed to find window for {file_name} within {timeout} seconds.", 500

    except Exception as e:
        return f"Failed to open {path}. Error: {e}", 500


@app.route("/setup/activate_window", methods=['POST'])
def activate_window():
    data = request.json
    window_name = data.get('window_name', None)
    if not window_name:
        return "window_name required", 400
    strict: bool = data.get("strict", False)  # compare case-sensitively and match the whole string
    by_class_name: bool = data.get("by_class", False)

    os_name = platform.system()

    if os_name == 'Windows':
        import pygetwindow as gw
        if by_class_name:
            return "Get window by class name is not supported on Windows currently.", 500
        windows: List[gw.Window] = gw.getWindowsWithTitle(window_name)

        window: Optional[gw.Window] = None
        if len(windows) == 0:
            return "Window {:} not found (empty results)".format(window_name), 404
        elif strict:
            for wnd in windows:
                if wnd.title == wnd:
                    window = wnd
            if window is None:
                return "Window {:} not found (strict mode).".format(window_name), 404
        else:
            window = windows[0]
        window.activate()

    elif os_name == 'Darwin':
        import pygetwindow as gw
        if by_class_name:
            return "Get window by class name is not supported on macOS currently.", 500
        # Find the VS Code window
        windows = gw.getWindowsWithTitle(window_name)

        window: Optional[gw.Window] = None
        if len(windows) == 0:
            return "Window {:} not found (empty results)".format(window_name), 404
        elif strict:
            for wnd in windows:
                if wnd.title == wnd:
                    window = wnd
            if window is None:
                return "Window {:} not found (strict mode).".format(window_name), 404
        else:
            window = windows[0]

        # Un-minimize the window and then bring it to the front
        window.unminimize()
        window.activate()

    elif os_name == 'Linux':
        # Attempt to activate VS Code window using wmctrl
        subprocess.run(["wmctrl"
                           , "-{:}{:}a".format("x" if by_class_name else ""
                                               , "F" if strict else ""
                                               )
                           , window_name
                        ]
                       )

    else:
        return f"Operating system {os_name} not supported.", 400

    return "Window activated successfully", 200


@app.route("/setup/close_window", methods=["POST"])
def close_window():
    data = request.json
    if "window_name" not in data:
        return "window_name required", 400
    window_name: str = data["window_name"]
    strict: bool = data.get("strict", False)  # compare case-sensitively and match the whole string
    by_class_name: bool = data.get("by_class", False)

    os_name: str = platform.system()
    if os_name == "Windows":
        import pygetwindow as gw

        if by_class_name:
            return "Get window by class name is not supported on Windows currently.", 500
        windows: List[gw.Window] = gw.getWindowsWithTitle(window_name)

        window: Optional[gw.Window] = None
        if len(windows) == 0:
            return "Window {:} not found (empty results)".format(window_name), 404
        elif strict:
            for wnd in windows:
                if wnd.title == wnd:
                    window = wnd
            if window is None:
                return "Window {:} not found (strict mode).".format(window_name), 404
        else:
            window = windows[0]
        window.close()
    elif os_name == "Linux":
        subprocess.run(["wmctrl"
                           , "-{:}{:}c".format("x" if by_class_name else ""
                                               , "F" if strict else ""
                                               )
                           , window_name
                        ]
                       )
    elif os_name == "Darwin":
        import pygetwindow as gw
        return "Currently not supported on macOS.", 500
    else:
        return "Not supported platform {:}".format(os_name), 500

    return "Window closed successfully.", 200


@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording_process
    if recording_process and recording_process.poll() is None:
        return jsonify({'status': 'error', 'message': 'Recording is already in progress.'}), 400

    # Clean up previous recording if it exists
    if os.path.exists(recording_path):
        try:
            os.remove(recording_path)
        except OSError as e:
            logger.error(f"Error removing old recording file: {e}")
            return jsonify({'status': 'error', 'message': f'Failed to remove old recording file: {e}'}), 500

    d = display.Display()
    screen_width = d.screen().width_in_pixels
    screen_height = d.screen().height_in_pixels

    start_command = f"ffmpeg -y -f x11grab -draw_mouse 1 -s {screen_width}x{screen_height} -i :0.0 -c:v libx264 -r 30 {recording_path}"

    # Use stderr=PIPE to capture potential errors from ffmpeg
    recording_process = subprocess.Popen(shlex.split(start_command),
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE,
                                         text=True  # To get stderr as string
                                         )

    # Wait a couple of seconds to see if ffmpeg starts successfully
    try:
        # Wait for 2 seconds. If ffmpeg exits within this time, it's an error.
        recording_process.wait(timeout=2)
        # If wait() returns, it means the process has terminated.
        error_output = recording_process.stderr.read()
        return jsonify({
            'status': 'error',
            'message': f'Failed to start recording. ffmpeg terminated unexpectedly. Error: {error_output}'
        }), 500
    except subprocess.TimeoutExpired:
        # This is the expected outcome: the process is still running after 2 seconds.
        return jsonify({'status': 'success', 'message': 'Started recording successfully.'})


@app.route('/end_recording', methods=['POST'])
def end_recording():
    global recording_process

    if not recording_process or recording_process.poll() is not None:
        recording_process = None  # Clean up stale process object
        return jsonify({'status': 'error', 'message': 'No recording in progress to stop.'}), 400

    error_output = ""
    try:
        # Send SIGINT for a graceful shutdown, allowing ffmpeg to finalize the file.
        recording_process.send_signal(signal.SIGINT)
        # Wait for ffmpeg to terminate. communicate() gets output and waits.
        _, error_output = recording_process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg did not respond to SIGINT, killing the process.")
        recording_process.kill()
        # After killing, communicate to get any remaining output.
        _, error_output = recording_process.communicate()
        recording_process = None
        return jsonify({
            'status': 'error',
            'message': f'Recording process was unresponsive and had to be killed. Stderr: {error_output}'
        }), 500

    recording_process = None  # Clear the process from global state

    # Check if the recording file was created and is not empty.
    if os.path.exists(recording_path) and os.path.getsize(recording_path) > 0:
        return send_file(recording_path, as_attachment=True)
    else:
        logger.error(f"Recording failed. The output file is missing or empty. ffmpeg stderr: {error_output}")
        return abort(500, description=f"Recording failed. The output file is missing or empty. ffmpeg stderr: {error_output}")


@app.route("/run_python", methods=['POST'])
def run_python():
    data = request.json
    code = data.get('code', None)

    if not code:
        return jsonify({'status': 'error', 'message': 'Code not supplied!'}), 400

    # Create a temporary file to save the Python code
    import tempfile
    import uuid

    # Generate unique filename
    temp_filename = f"/tmp/python_exec_{uuid.uuid4().hex}.py"

    try:
        # Write code to temporary file
        with open(temp_filename, 'w') as f:
            f.write(code)

        # Execute the file using subprocess to capture all output
        result = subprocess.run(
            ['/usr/bin/python3', temp_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30  # 30 second timeout
        )

        # Clean up the temporary file
        try:
            os.remove(temp_filename)
        except:
            pass  # Ignore cleanup errors

        # Prepare response
        output = result.stdout
        error_output = result.stderr

        # Combine output and errors if both exist
        combined_message = output
        if error_output:
            combined_message += ('\n' + error_output) if output else error_output

        # Determine status based on return code and errors
        if result.returncode != 0:
            status = 'error'
            if not error_output:
                # If no stderr but non-zero return code, add a generic error message
                error_output = f"Process exited with code {result.returncode}"
                combined_message = combined_message + '\n' + error_output if combined_message else error_output
        else:
            status = 'success'

        return jsonify({
            'status': status,
            'message': combined_message,
            'need_more': False,      # Not applicable for file execution
            'output': output,        # stdout only
            'error': error_output,   # stderr only
            'return_code': result.returncode
        })

    except subprocess.TimeoutExpired:
        # Clean up the temporary file on timeout
        try:
            os.remove(temp_filename)
        except:
            pass

        return jsonify({
            'status': 'error',
            'message': 'Execution timeout: Code took too long to execute',
            'error': 'TimeoutExpired',
            'need_more': False,
            'output': None,
        }), 500

    except Exception as e:
        # Clean up the temporary file on error
        try:
            os.remove(temp_filename)
        except:
            pass

        # Capture the exception details
        return jsonify({
            'status': 'error',
            'message': f'Execution error: {str(e)}',
            'error': traceback.format_exc(),
            'need_more': False,
            'output': None,
        }), 500


@app.route("/run_bash_script", methods=['POST'])
def run_bash_script():
    data = request.json
    script = data.get('script', None)
    timeout = data.get('timeout', 100)  # Default timeout of 30 seconds
    working_dir = data.get('working_dir', None)

    if not script:
        return jsonify({
            'status': 'error',
            'output': 'Script not supplied!',
            'error': "",  # Always empty as requested
            'returncode': -1
        }), 400

    # Expand user directory if provided
    if working_dir:
        working_dir = os.path.expanduser(working_dir)
        if not os.path.exists(working_dir):
            return jsonify({
                'status': 'error',
                'output': f'Working directory does not exist: {working_dir}',
                'error': "",  # Always empty as requested
                'returncode': -1
            }), 400

    # Create a temporary script file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp_file:
        if "#!/bin/bash" not in script:
            script = "#!/bin/bash\n\n" + script
        tmp_file.write(script)
        tmp_file_path = tmp_file.name

    try:
        # Make the script executable
        os.chmod(tmp_file_path, 0o755)

        # Execute the script
        if platform_name == "Windows":
            # On Windows, use Git Bash or WSL if available, otherwise cmd
            flags = subprocess.CREATE_NO_WINDOW
            # Try to use bash if available (Git Bash, WSL, etc.)
            result = subprocess.run(
                ['bash', tmp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                timeout=timeout,
                cwd=working_dir,
                creationflags=flags,
                shell=False
            )
        else:
            # On Unix-like systems, use bash directly
            flags = 0
            result = subprocess.run(
                ['/bin/bash', tmp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                timeout=timeout,
                cwd=working_dir,
                creationflags=flags,
                shell=False
            )

        # Log the command execution for trajectory recording
        # _append_event("BashScript",
        #               {"script": script, "output": result.stdout, "error": "", "returncode": result.returncode},
        #               ts=time.time())

        return jsonify({
            'status': 'success' if result.returncode == 0 else 'error',
            'output': result.stdout,  # Contains both stdout and stderr merged
            'error': "",  # Always empty as requested
            'returncode': result.returncode
        })

    except subprocess.TimeoutExpired:
        return jsonify({
            'status': 'error',
            'output': f'Script execution timed out after {timeout} seconds',
            'error': "",  # Always empty as requested
            'returncode': -1
        }), 500
    except FileNotFoundError:
        # Bash not found, try with sh
        try:
            result = subprocess.run(
                ['sh', tmp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                timeout=timeout,
                cwd=working_dir,
                shell=False
            )

            # _append_event("BashScript",
            #               {"script": script, "output": result.stdout, "error": "", "returncode": result.returncode},
            #               ts=time.time())

            return jsonify({
                'status': 'success' if result.returncode == 0 else 'error',
                'output': result.stdout,  # Contains both stdout and stderr merged
                'error': "",  # Always empty as requested
                'returncode': result.returncode,
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'output': f'Failed to execute script: {str(e)}',
                'error': "",  # Always empty as requested
                'returncode': -1
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'output': f'Failed to execute script: {str(e)}',
            'error': "",  # Always empty as requested
            'returncode': -1
        }), 500
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
