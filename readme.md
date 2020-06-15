in sensor folder:

run dataprocessing.py to generate data. Here, you can select three different types of data to create in the py file, which are original data, overlapping data and downsampled data.
first select  fn_num==1 to extract raw csv file from the xml, then you could switch to mode 2 or 3 to create overlapping or downsampled data based on this generated raw csv file
(processing.py is the function codes for the dataprocessing.py)

there are in total 14 rounds of data. Each round data contains n.xml (log file) and n.csv (ground truth special location)
n.csv formats as below:
 Time (relavent time that calculated by minus the first time value)+ lat1 + lng1 (these two values are directly from the App google map API) + x + y (these two are converted from lat&lng to utm) + lat +lng (these two are the relevent values which are conveinent for coordinate)