def download_history(pair, days, price_type="LAST_PRICE"):
    import requests
    import pandas as pd
    import time
    from datetime import datetime, timezone, timedelta

    API_URL = "https://api.sharkexchange.in/v1/market/klines"
    LIMIT = 1000

    print("\n" + "=" * 70)
    print("SHARK EXCHANGE HISTORICAL DOWNLOAD")
    print("=" * 70)

    now = datetime.now(timezone.utc)
    requested_start = now - timedelta(days=days)

    # Milliseconds
    current_end_ms = int(now.timestamp() * 1000)
    requested_start_ms = int(requested_start.timestamp() * 1000)

    all_candles = []
    page = 1

    while True:

        end_dt = datetime.fromtimestamp(
            current_end_ms / 1000,
            tz=timezone.utc
        )

        print(f"\nDownloading page {page} (through: {end_dt})")

        params = {
            "priceType": price_type
        }

        payload = {
            "pair": pair,
            "interval": "5m",
            "endTime": current_end_ms,
            "limit": LIMIT
        }

        try:
            response = requests.post(
                API_URL,
                params=params,
                json=payload,
                timeout=30
            )

        except Exception as e:
            raise RuntimeError(
                f"Shark API connection failed: {e}"
            )

        print("HTTP STATUS:", response.status_code)

        # Shark may return 200 or 201
        if response.status_code not in (200, 201):

            print("\nAPI RESPONSE:")
            print(response.text[:3000])

            raise RuntimeError(
                f"Shark API returned HTTP {response.status_code}"
            )

        # ---------------------------------------------------------
        # JSON PARSER
        # ---------------------------------------------------------

        try:
            raw = response.json()

        except Exception:

            print("\nCould not decode JSON.")
            print("RAW RESPONSE:")
            print(response.text[:3000])

            raise RuntimeError(
                "Shark API response was not valid JSON."
            )

        # ---------------------------------------------------------
        # HANDLE DIFFERENT RESPONSE WRAPPERS
        # ---------------------------------------------------------

        candles = None

        if isinstance(raw, list):
            candles = raw

        elif isinstance(raw, dict):

            possible_keys = [
                "data",
                "result",
                "rows",
                "candles",
                "klines"
            ]

            for key in possible_keys:

                value = raw.get(key)

                if isinstance(value, list):
                    candles = value
                    break

                if isinstance(value, dict):

                    for subkey in [
                        "data",
                        "rows",
                        "candles",
                        "klines",
                        "result"
                    ]:

                        subvalue = value.get(subkey)

                        if isinstance(subvalue, list):
                            candles = subvalue
                            break

                    if candles is not None:
                        break

        if candles is None:

            print("\nUNEXPECTED SHARK RESPONSE FORMAT:")
            print(str(raw)[:5000])

            raise RuntimeError(
                "Historical data could not be parsed."
            )

        if len(candles) == 0:
            print("No more candles returned.")
            break

        # ---------------------------------------------------------
        # NORMALIZE CANDLES
        # ---------------------------------------------------------

        normalized = []

        for candle in candles:

            if not isinstance(candle, dict):
                continue

            try:

                start_time = candle.get("startTime")

                if start_time is None:
                    continue

                # Convert timestamp safely
                if isinstance(start_time, str):

                    stripped = start_time.strip()

                    # Numeric string
                    if stripped.replace(".", "", 1).isdigit():
                        start_time = int(float(stripped))

                    else:
                        # ISO datetime
                        dt = pd.to_datetime(
                            stripped,
                            utc=True,
                            errors="coerce"
                        )

                        if pd.isna(dt):
                            continue

                        start_time = int(dt.timestamp() * 1000)

                elif isinstance(start_time, (int, float)):

                    start_time = int(start_time)

                    # seconds -> milliseconds
                    if start_time < 10_000_000_000:
                        start_time *= 1000

                else:
                    continue

                normalized.append({

                    "timestamp": start_time,

                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),

                    "volume": float(
                        candle.get("volume", 0) or 0
                    )
                })

            except Exception as e:

                print(
                    "Skipped malformed candle:",
                    str(candle)[:500],
                    "| Error:",
                    e
                )

        if not normalized:

            print("\nFirst API item:")
            print(str(candles[0])[:2000])

            raise RuntimeError(
                "API returned candles but none could be normalized."
            )

        print(
            f"Page {page}: "
            f"{len(normalized)} valid candles"
        )

        all_candles.extend(normalized)

        # ---------------------------------------------------------
        # FIND OLDEST CANDLE
        # ---------------------------------------------------------

        oldest_ms = min(
            candle["timestamp"]
            for candle in normalized
        )

        oldest_dt = datetime.fromtimestamp(
            oldest_ms / 1000,
            tz=timezone.utc
        )

        print("Oldest candle:", oldest_dt)

        # Requested history reached
        if oldest_ms <= requested_start_ms:

            print("\nRequested historical period reached.")
            break

        # Prevent infinite loop
        next_end_ms = oldest_ms - 1

        if next_end_ms >= current_end_ms:

            raise RuntimeError(
                "Pagination stopped moving backwards."
            )

        current_end_ms = next_end_ms
        page += 1

        # Stay comfortably under API rate limit
        time.sleep(0.30)

        # Safety
        if page > 500:

            raise RuntimeError(
                "Pagination safety limit reached."
            )

    # -------------------------------------------------------------
    # BUILD DATAFRAME
    # -------------------------------------------------------------

    if not all_candles:

        raise RuntimeError(
            "No historical candles downloaded."
        )

    df = pd.DataFrame(all_candles)

    # Remove duplicates
    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last"
    )

    df = df.sort_values("timestamp")

    # Only requested period
    df = df[
        df["timestamp"] >= requested_start_ms
    ].copy()

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    # -------------------------------------------------------------
    # REMOVE CURRENT OPEN 5M CANDLE
    # -------------------------------------------------------------

    current_5m_start = pd.Timestamp.now(
        tz="UTC"
    ).floor("5min")

    df = df[
        df["timestamp"] < current_5m_start
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print("Candles :", len(df))

    if len(df):

        print(
            "From    :",
            df["timestamp"].iloc[0]
        )

        print(
            "To      :",
            df["timestamp"].iloc[-1]
        )

    expected = days * 24 * 12

    print("Expected approx :", expected)

    coverage = (
        len(df) / expected * 100
        if expected
        else 0
    )

    print(
        f"Coverage        : {coverage:.2f}%"
    )

    if len(df) < 200:

        raise RuntimeError(
            f"Not enough candles downloaded: {len(df)}"
        )

    return df
