from __future__ import annotations

from fetch_hrrr_selected_messages import hrrr_surface_url


def test_hrrr_surface_url() -> None:
    url = hrrr_surface_url("https://noaa-hrrr-bdp-pds.s3.amazonaws.com/", "20250623", 18, 23)
    assert url == "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20250623/conus/hrrr.t18z.wrfsfcf23.grib2"
