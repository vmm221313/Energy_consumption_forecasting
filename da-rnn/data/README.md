file: kresit_main_2018-oct_to_2019-oct.csv
contains data of kresit main power meter. (fields available: Series, Time, Value)

Time: (format:YYYY-MM-DDTHH:mm:ssZ) is the start time of every 5 minute window.
Value: aggregated value for the time window specified by Time column
Series: (distinct values: "power_k_m", "min power_k_m", "max power_k_m", "count power_k_m", "energy_consumed power_k_m", "slot_energy_consumed power_k_m")

"power_k_m": corresponding values in Value column are Aggregated power consumption of every 5 min time window.
"min power_k_m": corresponding values in Value column are Min power consumption in that 5 min window. 
"max power_k_m": corresponding values in Value column are Max power consumption in that 5 min window.
"count power_k_m": corresponding values in Value column are count of power data in that 5 min window.
"energy_consumed power_k_m": corresponding values in Value column are cumulative energy consumption.
"slot_energy_consumed power_k_m": ignore these values.

Non-availability of data in SEIL MySQL database might have caused missing values.

To visualize the data use the following link:
https://seil.cse.iitb.ac.in/grafana/d/FoL9QzKZk/power-data-visualization?orgId=1&var-granularity=%25&var-power_sensor=power_k_m&var-power_cache=ignore_cache&from=1537549000406&to=1573285582823&panelId=2&fullscreen&theme=dark


