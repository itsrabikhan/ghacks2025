# See PwrPak7 manual for more information: https://docs.novatel.com/OEM7/Content/PDFs/OEM7_Commands_Logs_Manual.pdf
# This manual will be referenced in the code comments.

# IMPORTS

import os
import subprocess
import time
from matplotlib.widgets import Slider
import numpy as np
import scipy.ndimage
import pandas as pd
import simplekml
import json
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


# GLOBAL VARIABLES

with open("title.txt", "r", encoding="utf-8") as file:
    title = file.read()
data = []


# UTILITY CLASSES

class Colors:
    """
    This class contains ANSI escape codes for text colors and styles.
    """

    RED = "\u001b[31m"
    GREEN = "\u001b[32m"
    YELLOW = "\u001b[33m"
    BLUE = "\u001b[34m"
    MAGENTA = "\u001b[35m"
    CYAN = "\u001b[36m"
    WHITE = "\u001b[37m"
    GRAY = "\u001b[90m"
    RESET = "\u001b[0m"
    BOLD = "\u001b[1m"
    UNDERLINE = "\u001b[4m"

class Config:
    """
    This class contains the configuration for the program.
    The configuration is stored in a JSON file called config.json.
    """

    # Default configuration dictionary.
    default = {
        "google_earth_path": "SET_PATH_HERE"
    }

    def get(key: str = None) -> dict:
        """
        Get the configuration dictionary from the config file.

        Parameters:
            key (str): The key to get from the configuration dictionary. Default is None. If None, the entire dictionary is returned.

        Returns:
            object: The value of the key in the configuration dictionary. If key is None, the entire dictionary is returned.
        """

        # If no key is provided, return the entire dictionary.
        if key is None:
            # If the file does not exist, create it with the default configuration.
            try:
                with open("config.json", "r") as file:
                    return json.load(file)
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                with open("config.json", "w") as file:
                    json.dump(Config.default, file, indent=4)
                return Config.default
        else:
            # If a key is provided, return the value of that key.
            try:
                with open("config.json", "r") as file:
                    return json.load(file)[key]
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                # If the file does not exist or is invalid, create it with the default configuration.
                with open("config.json", "w") as file:
                    json.dump(Config.default, file, indent=4)
                return Config.default[key]
            except KeyError:
                # If the key does not exist in the default configuration, return None.
                if key in Config.default:
                    return Config.default[key]
                else:
                    return None


# CLASSES

class Header:
    """
    The Header class represents the header of an ASCII message.
    See Section 1.1 of the PwrPak7 manual for more information.
    """

    def __init__(
        self,
        sync: str,
        message: str,
        port: str,
        sequence: float,
        idle_time: float,
        time_status: str,
        week: int,
        seconds: float,
        status: str,
        reserved: str,
        version: str
    ):
        """
        Initialize a new Header object.

        Parameters:
            sync (str): The synchronization character.
            message (str): The message type.
            port (str): The port type.
            sequence (float): The sequence number.
            idle_time (float): The idle time.
            time_status (str): The time status.
            week (int): The week number.
            seconds (float): The seconds.
            status (str): The status.
            reserved (str): The reserved field.
            version (str): The version number

        Returns:
            Header: A new Header object.
        """

        self.sync = sync
        self.message = message
        self.port = port
        self.sequence = sequence
        self.idle_time = idle_time
        self.time_status = time_status
        self.week = week
        self.seconds = seconds
        self.status = status
        self.reserved = reserved
        self.version = version

    def parse(header: str) -> "Header":
        """
        Parse an ASCII header string into a Header object.
        This header string must follow the format, excluding the trailing semicolon, but including the leading sync character.

        Parameters:
            header (str): The ASCII header string.

        Returns:
            Header: A new Header object.

        Raises:
            ValueError: If the header string is invalid.
        """

        # Check if header starts with sync character.
        if header[0] != "#":
            raise ValueError("ASCII Header must start with #")

        # Slice off sync character.
        header = header[1:]

        # Split the data into a list of strings.
        header = header.split(",")

        # Return a new Header object.
        try:
            return Header(
                sync="#",
                message=header[0],
                port=header[1],
                sequence=float(header[2]),
                idle_time=float(header[3]),
                time_status=header[4],
                week=int(header[5]),
                seconds=float(header[6]),
                status=header[7],
                reserved=header[8],
                version=header[9]
            )
        except ValueError as e:
            raise ValueError(f"Error parsing ASCII Header: {e}")
        
    def __str__(self) -> str:
        """
        Return a string representation of the Header object.
        """

        return f"Header(sync={self.sync}, message={self.message}, port={self.port}, sequence={self.sequence}, idle_time={self.idle_time}, time_status={self.time_status}, week={self.week}, seconds={self.seconds}, status={self.status}, reserved={self.reserved}, version={self.version})"

class Entry:
    """
    The Entry class represents an entry in an ASCII message.
    This class is abstract and should not be instantiated directly, but should be subclassed.
    """

    def __init__(self, header: Header):
        """
        Initialize a new Entry object.
        Do not instantiate this class directly, but subclass it instead.

        Parameters:
            header (Header): The Header object for this entry.
        
        Returns:
            Entry: A new Entry object
        """

        self.header = header

class POS(Entry):
    """
    The POS class represents a position entry in an ASCII message.
    It is expected that the header message is of type BESTPOS (with a trailing A for ASCII), and will raise an error if it is not.
    See Section 3.21 of the PwrPak7 manual for more information.

    Keep in mind that this class mostly truncates the data to only include the fields necessary for the project (with some extras).
    """

    def __init__(
        self,
        header: Header,
        sol_stat: str,
        pos_type: str,
        lat: float,
        lon: float,
        alt: float,
        undulation: float,
        datum_id: str,
        lat_std: float,
        lon_std: float,
        alt_std: float,
        stn_id: str,
        # Ignore the rest of the fields.
    ):
        """
        Initialize a new POS object.

        Parameters:
            header (Header): The Header object for this entry.
            sol_stat (str): The solution status.
            pos_type (str): The position type.
            lat (float): The latitude.
            lon (float): The longitude.
            alt (float): The altitude.
            undulation (float): The undulation.
            datum_id (str): The datum ID.
            lat_std (float): The latitude standard deviation.
            lon_std (float): The longitude standard deviation.
            alt_std (float): The altitude standard deviation.
            stn_id (str): The station ID.

        Returns:
            POS: A new POS object.
        """

        super().__init__(header)
        self.sol_stat = sol_stat
        self.pos_type = pos_type
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.undulation = undulation
        self.datum_id = datum_id
        self.lat_std = lat_std
        self.lon_std = lon_std
        self.alt_std = alt_std
        self.stn_id = stn_id

    def parse(data: str) -> "POS":
        """
        Parse an ASCII data string into a POS object.
        This data string must be a full ASCII message, including the header and data sections, separated by a semicolon.

        Parameters:
            data (str): The ASCII data string.

        Returns:
            POS: A new POS object.

        Raises:
            ValueError: If the data string is invalid.
        """

        # Check if data has a header and data section.
        if data.count(";") != 1:
            raise ValueError("Data must contain a semicolon separator.")
    
        # Split the data into header and data sections.
        header, data = data.split(";", 1)

        # Parse the header.
        try:
            header = Header.parse(header)
        except ValueError as e:
            raise ValueError(f"Error parsing ASCII Header: {e}")

        # Check if header message is of type BESTPOSA.
        if header.message != "BESTPOSA":
            raise ValueError("Pos object header message must be BESTPOSA.") 

        # Split the data into a list of strings.
        data = data.split(",")

        # Return a new Pos object.
        try:
            return POS(
                header=header,
                sol_stat=data[0],
                pos_type=data[1],
                lat=float(data[2]),
                lon=float(data[3]),
                alt=float(data[4]),
                undulation=float(data[5]),
                datum_id=data[6],
                lat_std=float(data[7]),
                lon_std=float(data[8]),
                alt_std=float(data[9]),
                stn_id=data[10]
            )
        except ValueError as e:
            raise ValueError(f"Error parsing ASCII Position Data: {e}")

    def __str__(self) -> str:
        return f"Pos(header={self.header}, sol_stat={self.sol_stat}, pos_type={self.pos_type}, lat={self.lat}, lon={self.lon}, alt={self.alt}, undulation={self.undulation}, datum_id={self.datum_id}, lat_std={self.lat_std}, lon_std={self.lon_std}, alt_std={self.alt_std}, stn_id={self.stn_id})"
    
class XYZ(Entry):
    """
    The XYZ class represents a position and velocity entry in an ASCII message.
    It is expected that the header message is of type BESTXYZ (with a trailing A for ASCII), and will raise an error if it is not.
    See Section 3.25 of the PwrPak7 manual for more information.

    Keep in mind that this class mostly truncates the data to only include the fields necessary for the project (with some extras).
    """

    def __init__(
        self,
        header: Header,
        sol_stat: str,
        pos_type: str,
        x: float,
        y: float,
        z: float,
        x_std: float,
        y_std: float,
        z_std: float,
        v_sol_stat: str,
        vel_type: str,
        vx: float,
        vy: float,
        vz: float,
        vx_std: float,
        vy_std: float,
        vz_std: float,
        # Ignore the rest of the fields.
    ):
        """
        Initialize a new XYZ object.

        Parameters:
            header (Header): The Header object for this entry.
            sol_stat (str): The solution status.
            pos_type (str): The position type.
            x (float): The X coordinate.
            y (float): The Y coordinate.
            z (float): The Z coordinate.
            x_std (float): The X standard deviation.
            y_std (float): The Y standard deviation.
            z_std (float): The Z standard deviation.
            v_sol_stat (str): The velocity solution status.
            vel_type (str): The velocity type.
            vx (float): The X velocity.
            vy (float): The Y velocity.
            vz (float): The Z velocity.
            vx_std (float): The X velocity standard deviation.
            vy_std (float): The Y velocity standard deviation.
            vz_std (float): The Z velocity standard deviation.

        Returns:
            XYZ: A new XYZ object.
        """

        super().__init__(header)
        self.sol_stat = sol_stat
        self.pos_type = pos_type
        self.x = x
        self.y = y
        self.z = z
        self.x_std = x_std
        self.y_std = y_std
        self.z_std = z_std
        self.v_sol_stat = v_sol_stat
        self.vel_type = vel_type
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.vx_std = vx_std
        self.vy_std = vy_std
        self.vz_std = vz_std

    def parse(data: str) -> "XYZ":
        """
        Parse an ASCII data string into a XYZ object.
        This data string must be a full ASCII message, including the header and data sections, separated by a semicolon.

        Parameters:
            data (str): The ASCII data string.

        Returns:
            XYZ: A new XYZ object.

        Raises:
            ValueError: If the data string is invalid.
        """

        # Check if data has a header and data section.
        if data.count(";") != 1:
            raise ValueError("Data must contain a semicolon separator.")
    
        # Split the data into header and data sections.
        header, data = data.split(";", 1)

        # Parse the header.
        try:
            header = Header.parse(header)
        except ValueError as e:
            raise ValueError(f"Error parsing ASCII Header: {e}")

        # Check if header message is of type BESTXYZA.
        if header.message != "BESTXYZA":
            raise ValueError("XYZ object header message must be BESTXYZA.") 

        # Split the data into a list of strings.
        data = data.split(",")

        # Return a new XYZ object.
        try:
            return XYZ(
                header=header,
                sol_stat=data[0],
                pos_type=data[1],
                x=float(data[2]),
                y=float(data[3]),
                z=float(data[4]),
                x_std=float(data[5]),
                y_std=float(data[6]),
                z_std=float(data[7]),
                v_sol_stat=data[8],
                vel_type=data[9],
                vx=float(data[10]),
                vy=float(data[11]),
                vz=float(data[12]),
                vx_std=float(data[13]),
                vy_std=float(data[14]),
                vz_std=float(data[15])
            )
        except ValueError as e:
            raise ValueError(f"Error parsing ASCII Position Data: {e}")
    
    def __str__(self) -> str:
        return f"XYZ(header={self.header}, sol_stat={self.sol_stat}, pos_type={self.pos_type}, x={self.x}, y={self.y}, z={self.z}, x_std={self.x_std}, y_std={self.y_std}, z_std={self.z_std}, v_sol_stat={self.v_sol_stat}, vel_type={self.vel_type}, vx={self.vx}, vy={self.vy}, vz={self.vz}, vx_std={self.vx_std}, vy_std={self.vy_std}, vz_std={self.vz_std})"


# UTILITY FUNCTIONS

def clear() -> None:
    """
    Clear the console screen.
    """

    print("\033[H\033[J")

def pause(newlines: int = 1) -> None:
    """
    Pause the program until the user presses Enter.

    Parameters:
        newlines (int): The number of newlines to print before the prompt. Default is 1.

    Returns:
        None
    """

    # Print newlines and prompt the user to press the enter key.
    fprint(("\n" * newlines) + "Press ENTER to continue.", end="")
    input()

def fprint(*args, end: str = "\n") -> None:
    """
    Print function with built-in color reset for convenience.
    This is to be used instead of the built-in print function because of its automatic color reset.

    Parameters:
        *args (any): Any number of arguments to print, can be any type.
        end (str): The string to append to the end of the printed string. Default is a newline.
    """

    # Altered print statement to include color reset.
    print(str(*args) + Colors.RESET, end=end)

def gaussian_smooth(data: pd.DataFrame, std_values: pd.DataFrame, factor: float = None) -> pd.DataFrame:
    """
    Apply Gaussian smoothing to the given DataFrame using row-wise standard deviations as sigma.
    This function will be used to smooth the data given by the device.
    
    Parameters:
        data (pd.DataFrame): The DataFrame containing numerical data.
        std_values (pd.DataFrame): DataFrame with the same shape as `data` containing standard deviation values per row.
        factor (float): The factor to multiply the standard deviation values by. Default is 3.
    
    Returns:
        pd.DataFrame: Smoothed DataFrame.
    """

    # Default factor is 3.
    if factor is None:
        factor = 3

    # Copy the data to avoid modifying the original.
    denoised_data = data.copy()
    
    # Apply Gaussian filter to each column.
    for column in data.columns:
        if column in std_values.columns:
            # Use scipy.ndimage.gaussian_filter1d to apply the Gaussian filter.
            denoised_data[column] = [
                scipy.ndimage.gaussian_filter1d(data[column].values, sigma=row_sigma * factor)[i]
                for i, row_sigma in enumerate(std_values[column].values)
            ]

    # Return the denoised data.
    return denoised_data

def smooth_points(data: list[POS], factor: float = None) -> list[tuple]:
    """
    Convenience function to smooth the given list of POS objects using the Gaussian filter.
    This is mainly used to reduce code duplication and improve readability.

    Parameters:
        data (list[POS]): The list of POS objects to smooth.
        factor (float): The factor to multiply the standard deviation values by. Default is None, meaning the default factor in `gaussian_smooth` is used.

    Returns:
        list[tuple]: A list of tuples containing the smoothed data. The tuple format is (latitude, longitude, altitude).
    """

    # Convert the data to a DataFrame.
    coordinates = pd.DataFrame({
        "Latitude": [entry.lat for entry in data if isinstance(entry, POS)],
        "Longitude": [entry.lon for entry in data if isinstance(entry, POS)],
        "Altitude": [entry.alt for entry in data if isinstance(entry, POS)],
    })
    deviations = pd.DataFrame({
        "Latitude": [entry.lat_std for entry in data if isinstance(entry, POS)],
        "Longitude": [entry.lon_std for entry in data if isinstance(entry, POS)],
        "Altitude": [entry.alt_std for entry in data if isinstance(entry, POS)],
    })

    # Denoise the data.
    denoised = gaussian_smooth(coordinates, deviations)
    return [(row["Latitude"], row["Longitude"], row["Altitude"]) for _, row in denoised.iterrows()]

def haversine(lat1, lon1, lat2, lon2) -> float:
    """
    Calculate the great-circle distance between two points on the Earth's surface.

    Parameters:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: The distance between the two points in meters.
    """

    # Use the Haversine formula to calculate the distance between two points on the Earth's surface.
    # The radius of the Earth is 6371 km.
    radius = 6371000
    # Convert latitudes and longitudes to radians.
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Haversine formula.
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return radius * c

def get_flattened_altitudes(altitudes: list[float], factor: int) -> list[float]:
    """
    Adjusts altitudes based on the flattening factor in percent.

    Parameters:
        altitudes (list[float]): A list of altitudes to adjust.
        factor (int): The flattening factor in percent.

    Returns:
        list[float]: A list of adjusted altitudes.
    """

    # Calculate the mean altitude.
    mean_altitude = np.mean(altitudes)

    return [(factor / 100) * alt + (1 - factor / 100) * mean_altitude for alt in altitudes]


# MENU FUNCTIONS

def print_title() -> None:
    """
    Print the program title.
    """
    fprint(Colors.BOLD + Colors.MAGENTA + title)


def main_menu() -> None:
    """
    Display the main menu.
    """

    while True:
        clear()
        print_title()
        fprint(Colors.BOLD + "MAIN MENU")
        fprint("  1. Render on Earth")
        fprint("  2. Render in 3D")
        fprint("  3. Plot Speed")
        fprint("  9. Help")
        fprint("  0. Exit")

        choice = input("\n> ")
        match choice:
            case "1":
                render_on_earth_menu()
            case "2":
                render_3d()
            case "3":
                plot_speed()
            case "9":
                help_menu()
            case "0":
                goodbye()
            case _:
                fprint(Colors.RED + "Invalid choice. Please try again.")
                pause()

def render_on_earth_menu() -> None:
    """
    Display the Google Earth rendering menu.
    """

    settings = {
        "raw": False,
        "smoothed": False,
        "line": True,
        "endpoints": True,
    }

    while True:
        clear()
        print_title()
        fprint(Colors.BOLD + "RENDER EARTH")
        fprint("  1. Raw Points - " + (Colors.GREEN + "ON" if settings["raw"] else Colors.RED + "OFF"))
        fprint("  2. Smoothed Points - " + (Colors.GREEN + "ON" if settings["smoothed"] else Colors.RED + "OFF"))
        fprint("  3. Line - " + (Colors.GREEN + "ON" if settings["line"] else Colors.RED + "OFF"))
        fprint("  4. Endpoints - " + (Colors.GREEN + "ON" if settings["endpoints"] else Colors.RED + "OFF"))
        fprint("  5. Confirm")
        fprint("  9. Back")
        fprint("  0. Exit")

        choice = input("\n> ")
        match choice:
            case "1":
                settings["raw"] = not settings["raw"]
            case "2":
                settings["smoothed"] = not settings["smoothed"]
            case "3":
                settings["line"] = not settings["line"]
            case "4":
                settings["endpoints"] = not settings["endpoints"]
            case "5":
                if not any(settings.values()):
                    fprint(Colors.RED + "Please enable at least one option.")
                    pause()
                    continue
                render_earth(settings)
                return
            case "9":
                return
            case "0":
                goodbye()
            case _:
                fprint(Colors.RED + "Invalid choice. Please try again.")
                pause()

def render_earth(settings: dict) -> None:
    """
    Generate and render KML files in Google Earth.
    Requires the Google Earth Pro application to be installed on the system.

    Parameters:
        settings (dict): A dictionary containing the settings for the KML file.
    """

    # Ensure that directory exists.
    if not os.path.exists("kml"):
        os.mkdir("kml")

    # Assume settings file is valid.
    # All KMLs are separated since the user can control the layers visible in Google Earth Pro.
    
    # Raw data KML.
    if settings["raw"]:
        raw = simplekml.Kml()
        # Index is used as time, since the data is taken at an interval of 1 second.
        for index, entry in enumerate(data):
            # Only add POS entries. Ignore XYZ entries.
            if isinstance(entry, POS):
                point = raw.newpoint(coords=[(entry.lon, entry.lat, entry.alt)])
                point.description = f"""
                    Time: {index}s

                    Latitude: {entry.lat}
                    Longitude: {entry.lon}
                    Altitude: {entry.alt}m
                """
                point.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        raw.save("kml/raw.kml")

    # Smoothed data KML.
    smoothed = smooth_points(data) # Smoothed is generated regardless, since it is also used for the line and endpoints.
    if settings["smoothed"]:
        gauss = simplekml.Kml()
        for index, entry in enumerate(smoothed):
            point = gauss.newpoint(coords=[(entry[1], entry[0], entry[2])])
            point.description = f"""
                Time: {index}s

                Latitude: {entry[0]}
                Longitude: {entry[1]}
                Altitude: {entry[2]}m
            """
            point.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        gauss.save("kml/smoothed.kml")

    # Line KML.
    if settings["line"]:
        line_kml = simplekml.Kml()
        line = line_kml.newlinestring(name="Path")
        line.coords = [(lon, lat, alt) for lat, lon, alt in smoothed]
        line.style.linestyle.color = simplekml.Color.red
        line.style.linestyle.width = 3
        line_kml.save("kml/line.kml")

    # Endpoints KML.
    if settings["endpoints"]:
        endpoints = simplekml.Kml()

        # Start point.
        start_lat, start_lon, start_alt = smoothed[0]
        start_point = endpoints.newpoint(name="Start Point", coords=[(start_lon, start_lat, start_alt)])
        start_point.description = f"""
            Start Point

            Latitude: {start_lat}
            Longitude: {start_lon}
            Altitude: {start_alt}m
        """
        start_point.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/ltblu-circle.png"

        # End point.
        end_lat, end_lon, end_alt = smoothed[-1]
        end_point = endpoints.newpoint(name="End Point", coords=[(end_lon, end_lat, end_alt)])
        end_point.description = f"""
            End Point

            Latitude: {end_lat}
            Longitude: {end_lon}
            Altitude: {end_alt}m
        """
        end_point.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"
        endpoints.save("kml/endpoints.kml")

    # Get Google Earth Pro path from config file.
    google_earth_path = Config.get("google_earth_path")
    if google_earth_path is None or not os.path.exists(google_earth_path):
        fprint(Colors.RED + "Google Earth Pro path not set or invalid. Please set the path in the config.json file.")
        pause()
        return
    
    # Create list for subprocess.run.
    array = [google_earth_path]
    for key, value in settings.items():
        if value:
            array.append(os.path.join(os.getcwd(), f"kml/{key}.kml"))

    # Open Google Earth Pro with the generated KML files.
    fprint(Colors.GREEN + "Opening Google Earth Pro... Please wait.")
    time.sleep(2) # Without using asyncio, this is the best way to ensure the files are saved before opening.
    subprocess.run(array, shell=True)
    pause()

def render_3d() -> None:
    """
    Render the 3D plot.
    """

    clear()
    print_title()
    fprint(Colors.BOLD + "3D PLOT")
    fprint("Rendering 3D plot... Please wait.")

    # Extract all available position data with latitude, longitude, and altitude.
    coordinates = [(entry.lat, entry.lon, entry.alt) for entry in data if isinstance(entry, POS)]

    # Extract all available velocity data with direction and standard deviation.
    velocities = [(entry.vx, entry.vy, entry.vz, entry.vx_std, entry.vy_std, entry.vz_std) for entry in data if isinstance(entry, XYZ)]

    # We assume that the data is in order and that the length of coordinates and velocities is the same.
    if len(coordinates) != len(velocities):
        fprint(Colors.RED + "Data is not in order or is missing. Please ensure that the data is complete.")
        pause()
        return

    # Compute velocity magnitudes.
    velocity_magnitudes = [np.sqrt(vx**2 + vy**2 + vz**2) for vx, vy, vz, _, _, _ in velocities]

    # Extract latitude, longitude, and altitude.
    latitudes = [coord[0] for coord in coordinates]
    longitudes = [coord[1] for coord in coordinates]
    altitudes = [coord[2] for coord in coordinates]

    current_latitudes = latitudes.copy()
    current_longitudes = longitudes.copy()
    current_altitudes = altitudes.copy()

    # Create the figure and 3D axis.
    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection='3d')

    # Plot the path taken.
    path_plot, = axis.plot(current_longitudes, current_latitudes, current_altitudes, marker="o", linestyle="-", color="blue", label="Path Taken")

    # Initial highlighted point (default at time = 0).
    highlight, = axis.plot([current_longitudes[0]], [current_latitudes[0]], [current_altitudes[0]], marker="o", color="red", markersize=10, label="Selected Point")

    # Create info box for displaying data.
    info_box = figure.add_axes([0.02, 0.75, 0.05, 0.15])
    info_box.axis("off")

    # Labels and title.
    axis.set_xlabel("Longitudinal Distance")
    axis.set_ylabel("Latitudinal Distance")
    axis.set_zlabel("Altitude")
    # Remove ticks for better visibility and to reduce clutter and confusion.
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.set_title("Interactive Path Visualization")
    axis.legend()

    # Create time slider.
    axis_time_slider = plt.axes([0.2, 0.02, 0.65, 0.03])
    time_slider = Slider(axis_time_slider, "Time (s)", 0, len(coordinates) - 1, valinit=0, valstep=1)

    # Create altitude flatten slider. 0% is the original flawed data, 100% is fully flattened.
    axis_flatten_slider = plt.axes([0.2, 0.07, 0.65, 0.03])
    flatten_slider = Slider(axis_flatten_slider, "Flattening %", 0, 100, valinit=0, valstep=1)

    # Create smoothing slider. This is used to adjust the smoothing factor.
    axis_smooth_slider = plt.axes([0.2, 0.12, 0.65, 0.03])
    smooth_slider = Slider(axis_smooth_slider, "Denoising Factor", 0, 3, valinit=0, valstep=0.1)

    # NOTE: This function is nested to allow access to all variables and keep code structure clean.
    def update(_) -> None:
        """
        Update the plot when the processing sliders change.

        Parameters:
            _: Unused parameter, required for Slider.
        """

        # Update path and highlighted point.
        if smooth_slider.val != 0:
            coordinates = pd.DataFrame({
                "Latitude": latitudes,
                "Longitude": longitudes,
                "Altitude": altitudes,
            })

            deviations = pd.DataFrame({
                "Latitude": [entry.lat_std for entry in data if isinstance(entry, POS)],
                "Longitude": [entry.lon_std for entry in data if isinstance(entry, POS)],
                "Altitude": [entry.alt_std for entry in data if isinstance(entry, POS)],
            })

            smoothed = gaussian_smooth(coordinates, deviations, smooth_slider.val)
            current_latitudes = smoothed["Latitude"].values
            current_longitudes = smoothed["Longitude"].values
            current_altitudes = smoothed["Altitude"].values
        else:
            current_latitudes = latitudes.copy()
            current_longitudes = longitudes.copy()
            current_altitudes = altitudes.copy()

        current_altitudes = get_flattened_altitudes(current_altitudes, 100 - flatten_slider.val)    

        path_plot.set_data(current_longitudes, current_latitudes)
        path_plot.set_3d_properties(current_altitudes)

        # Ensure synchronization with time slider.
        time = int(time_slider.val)

        highlight.set_data([current_longitudes[int(time_slider.val)]], [current_latitudes[int(time_slider.val)]])
        highlight.set_3d_properties([current_altitudes[int(time_slider.val)]])

        # Compute distances between points.
        new_coordinates = [(lat, lon) for lat, lon in zip(current_latitudes, current_longitudes)]
        distances = [0]
        for i in range(1, len(new_coordinates)):
            distance = haversine(new_coordinates[i - 1][0], new_coordinates[i - 1][1], new_coordinates[i][0], new_coordinates[i][1])
            # Compute cumulative distance.
            distances.append(distances[-1] + distance)

        speed = velocity_magnitudes[time]
        distance_traveled = distances[time]

        # Update info box.
        info_box.clear()
        info_box.axis("off")
        info_box.text(0.1, 0.8, f"Time: {time}s", fontsize=10, fontweight="bold")
        info_box.text(0.1, 0.6, f"Latitude: {current_latitudes[time]:.6f}{' (denoised)' if smooth_slider.val != 0 else ''}", fontsize=10)
        info_box.text(0.1, 0.4, f"Longitude: {current_longitudes[time]:.6f}{' (denoised)' if smooth_slider.val != 0 else ''}", fontsize=10)
        info_box.text(0.1, 0.2, f"Altitude: {current_altitudes[time]:.2f}m{' (flattened)' if flatten_slider.val != 0 else ''}", fontsize=10)
        info_box.text(0.1, 0.0, f"Speed: {speed:.3f} m/s", fontsize=10)
        info_box.text(0.1, -0.2, f"Distance Travelled: {distance_traveled:.2f}m{' (noisy)' if smooth_slider.val == 0 else ''}", fontsize=10)

        # Update plot.
        figure.canvas.draw_idle()        

    # Attach the same function to the sliders.
    time_slider.on_changed(update)
    flatten_slider.on_changed(update)
    smooth_slider.on_changed(update)

    # Finally, show the plot.
    plt.show()
    pause()

def plot_speed() -> None:
    """
    Plot the speed of the device over time.
    """

    clear()
    print_title()
    fprint(Colors.BOLD + "SPEED PLOT")
    fprint("Rendering speed plot... Please wait.")

    # Extract all available velocity data with direction and standard deviation.
    velocities = [(entry.vx, entry.vy, entry.vz) for entry in data if isinstance(entry, XYZ)]
    speeds = [np.sqrt(vx**2 + vy**2 + vz**2) for vx, vy, vz in velocities]

    # Generate time steps.
    time_steps = np.arange(len(velocities))

    # Create the figure and axis.
    figure, axis = plt.subplots(figsize=(12, 9))
    line = axis.plot(time_steps, speeds, label="Speed", color="blue", linewidth=2)[0]
    axis.set_title("Speed vs. Time")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Speed (m/s)")
    axis.legend()
    axis.grid(True)

    # Add smoothing factor slider.
    axis_smooth_slider = plt.axes([0.2, 0.02, 0.65, 0.03])
    smooth_slider = Slider(axis_smooth_slider, "Denoising Factor", 0, 2, valinit=0, valstep=0.1)

    def update(_) -> None:
        """
        Update the plot when the smoothing factor slider changes.

        Parameters:
            _: Unused parameter, required for Slider.
        """

        # Smooth the data using the Gaussian filter.
        if smooth_slider.val != 0:
            # This does not take the standard deviations into account.
            # This decision was made to simplify the process and reduce complexity.
            current_speeds = scipy.ndimage.gaussian_filter1d(speeds, sigma=smooth_slider.val)
        else:
            current_speeds = speeds
        
        # Update plot.
        line.set_ydata(current_speeds)
        figure.canvas.draw_idle()

    # Attach the update function to the slider.
    smooth_slider.on_changed(update)

    # Finally, show the plot.
    plt.show()
    pause()


def help_menu() -> None:
    """
    Display the help menu.
    """

    clear()
    print_title()
    fprint(Colors.BOLD + "HELP MENU -> OVERVIEW")
    fprint("This program reads data from a file and allows you to visualize it in Google Earth Pro or a 3D plot.")
    pause()

    clear()
    print_title()
    fprint(Colors.BOLD + "HELP MENU -> RENDER ON EARTH")
    fprint("You may choose to render the raw data, smoothed data, a line, and endpoints in Google Earth Pro.")
    fprint("For this option, you must have Google Earth Pro installed on your system.")
    fprint("You must also set the path to the Google Earth Pro executable in the config.json file.")
    fprint("\nThere is also a known issue with Google Earth Pro not loading all selected options,")
    fprint("This is a known issue, and has been traced back to Google Earth Pro itself.")
    pause()

    clear()
    print_title()
    fprint(Colors.BOLD + "HELP MENU -> RENDER 3D")
    fprint("You may also render the data in a 3D plot.")
    fprint("This plot allows you to visualize the path taken, the velocity, and the distance traveled.\n")
    fprint("You may change the denoising factor, to adjust the smoothness of the path to reduce noise.")
    fprint("You can also flatten the altitude to see the path more clearly.")
    fprint("You may use the time slider to view the data at different points in time.\n")
    fprint("Keep in mind that the denoising factor and flattening may affect the accuracy of the time-based data in the top left.")
    pause()

    clear()
    print_title()
    fprint(Colors.BOLD + "HELP MENU -> PLOT SPEED")
    fprint("You may also plot the speed of the device over time.")
    fprint("This plot shows the speed of the device at each time step.")
    fprint("You may adjust the denoising factor to reduce noise in the plot.")
    pause()


def goodbye() -> None:
    """
    Print a goodbye message and exit the program.
    """

    fprint(Colors.GREEN + "Goodbye!")
    exit()


# MAIN FUNCTION

def main() -> None:
    # Generate configuration file if it does not exist.
    try:
        Config.get()
    except Exception as e:
        fprint(Colors.RED + f"Error loading configuration: {e}")
        goodbye()

    # Open a file dialog to select the log file.
    fprint("Opening file dialog to select the log file...")
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select the log",
        filetypes=(
            ("Text files", "*.txt"),
            ("All files", "*.*")
        )
    )

    # Load the file and parse the data.
    fprint(f"Loading file '{filename}'...")
    try:
        raw = open(filename, "r").readlines()
    except FileNotFoundError:
        # This technically should not happen, but it is better to be safe.
        fprint(Colors.RED + "File not found. Please try again.")
        goodbye()
    except ImportError:
        fprint(Colors.RED + "An error occurred while loading the log file. Please ensure that the file is valid.")
        goodbye()
    except Exception as e:
        fprint(Colors.RED + f"An error occurred while loading the log file: {e}")
        goodbye()

    # Parse data into objects.
    for line in raw:
        try:
            if line.startswith("#BESTPOSA"):
                pos = POS.parse(line)
                data.append(pos)
            elif line.startswith("#BESTXYZA"):
                xyz = XYZ.parse(line)
                data.append(xyz)
            else:
                fprint(Colors.YELLOW + f"Skipping line: {line}")
        except ValueError as e:
            fprint(Colors.RED + f"Error parsing line: {e}")

    #pause() # Uncomment to pause after parsing data for debugging.

    # Check if data is empty.
    if len(data) == 0:
        fprint(Colors.RED + "No data found in the log file. Please ensure that the file is valid.")
        goodbye()

    # Launch main menu.
    main_menu()


# ENTRY POINT

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fprint(Colors.RED + "\nProgram terminated by user.")
        goodbye()