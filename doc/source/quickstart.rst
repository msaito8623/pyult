Quickstart
==========

Installation
------------

First, you need to install *pyult*. Please look at :doc:`installation` for more details for the installation of *pyult*


Necessary files: Exported files by Articulate Assistant Advanced
----------------------------------------------------------------

*pyult* provides a series of functions to preprocess ultrasound-related files exported by `Articulate Assistant Advanced (AAA) <http://www.articulateinstruments.com/aaa/>`_, which is a product by Articulate Instruments Ltd [1]_.

Exported files by AAA for each recording usually consist of the main ultrasound file with the extension *.ult*, a parameter file (which usually ends with *...US.txt*), a meta-information text file containing information such as prompts (words/phrases displayed on a screen during recording) with the extension *.txt*, and an audio file (*.wav*).

Additionally, TextGrid files may be necessary, if you would like to include segment-information in the dataframes produced by *pyult*. TextGrid files can be produced by using forced-alignment programs.


Generate dataframes and pictures automatically from ultrasound files
--------------------------------------------------------------------

Basic Use
^^^^^^^^^

*pyult* can generate dataframes and pictures from ultrasound recordings (exported files) by the following command:

.. code:: bash

    python -m pyult

This command requires at least two arguments: path to the target directory and what task the script should do. They can be provided by -d (--directory) and -t (--task) options as below:

.. code:: bash
    python -m pyult -d /foo/bar -t df

For tasks, "df", "raw", "squ", "fan" and "video" are available for now. "df" produces dataframes for each recording. "raw", "squ", and "fan" produces images in the raw rectangle, square, and fan-shapes for each. "video" produces videos for each recording.

The target directory must have all the necessary files with corresponding names, i.e. the main ultrasound files (xxx.ult), parameter files (xxxUS.txt), meta-information files (xxx.txt), and audio files (xxx_Track0.wav) for each recording. "US" of parameter files and track numbers of audio files are ignored. Therefore, the target directory should have such a structure as below:

::

    session_01
    ├── recording_01.txt
    ├── recording_01US.txt
    ├── recording_01.ult
    ├── recording_01_Track0.wav
    ├── recording_02.txt
    ├── recording_02US.txt
    ├── recording_02.ult
    └── recording_02_Track0.wav









----

.. [1] Articulate Instruments Ltd. (2012). Articulate Assistant Advanced User Guide: Version 2.14. Edinburgh, UK: Articulate Instruments Ltd.
