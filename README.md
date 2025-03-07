# Replay

Have you ever walked around carrying a bunch of GNSS equipment logging data, then forgotten where you were 20 seconds ago? So have we! All jokes aside, Replay aims to streamline processing geospatial data and easily convert it into human-readable and interpretable data in various forms. We were inspired by many movie style interfaces, that allow characters to easily view data in such an easy way, we thought we could implement this into our project! This led us to using not one, but TWO 3-dimensional representations for our geospatial data.

Replay is a project developed for the G-Hacks hackathon, where it won second place. The tool reads GNSS receiver output files and creates visualizations using Google Earth and Matplotlib. It is designed to facilitate the analysis of location data through clear and interactive visual outputs.

## Overview

Replay allows its users to easily interpret and process geospatial data collected by the PwrPak7. Our approach ensures that the user is able to easily read and interpret all data, by giving the user as much interactive data as possible. The days of plaintext data reporting are long gone, and so are simple and basic graphs. Viewing data in an interactive and dynamic way is extremely important, and we believe that it is the future.

## Installation and Setup

Clone the repository and install the necessary Python dependencies using pip. The code requires Python 3 and the packages in `requirements.txt` for GNSS data parsing and Google Earth integration. Configuration details and installation commands are provided in the project itself. For Google Earth visualization, Google Earth Pro must be installed and its installation path must be inputted into the config file (generated on run).

## Usage

Run the main script with your GNSS data file as input. The program reads the file, parses the data, and can open visualization windows for Google Earth or Matplotlib. Adjustments to the code may be made for different data formats or custom visual outputs.

## Contribution and License

Contributions to improve data handling, visualization options, or integration features are welcome. The project is open for collaboration under the MIT License.
This project was simply for the hackathon, and may not be updated. However, you are encouraged to check it out and look into the way data is handled from the PwrPak7 receiver.
