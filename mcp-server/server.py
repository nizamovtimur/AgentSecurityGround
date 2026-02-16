from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import os

CSV_FILE = "bookings.csv"

# Инициализация MCP сервера
mcp = FastMCP(
    name="KiteClubBookings",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)

class Booking(BaseModel):
    name: str
    datetime: str
    notes: str
    contact: str

class UpdateBooking(BaseModel):
    id: int
    name: str | None = None
    datetime: str | None = None
    notes: str | None = None
    contact: str | None = None


# Вспомогательные функции для работы с CSV
def load_bookings() -> pd.DataFrame:
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["id", "name", "datetime", "notes", "contact", "created_at"])
    return df

def save_bookings(df: pd.DataFrame):
    df.to_csv(CSV_FILE, index=False)


# Tool: list all bookings
@mcp.tool()
def list_bookings() -> list[dict]:
    df = load_bookings()
    return df.to_dict(orient="records")


# Tool: create a new booking
@mcp.tool()
def create_booking(booking: Booking) -> dict:
    df = load_bookings()
    new_id = df["id"].max() + 1 if not df.empty else 1
    record = {
        "id": new_id,
        "name": booking.name,
        "datetime": booking.datetime,
        "notes": booking.notes,
        "contact": booking.contact,
        "created_at": datetime.now().isoformat(),
    }
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    save_bookings(df)
    return {
        "status": "success",
        "message": f"Booking created with id {new_id}",
        "booking": record,
    }


# Tool: update existing booking
@mcp.tool()
def update_booking(update: UpdateBooking) -> dict:
    df = load_bookings()
    if update.id in df["id"].values:
        idx = df.index[df["id"] == update.id][0]
        if update.name:
            df.at[idx, "name"] = update.name
        if update.datetime:
            df.at[idx, "datetime"] = update.datetime
        if update.notes:
            df.at[idx, "notes"] = update.notes
        if update.contact:
            df.at[idx, "contact"] = update.contact
        save_bookings(df)
        return {"status": "success", "message": "Booking updated", "booking": df.loc[idx].to_dict()}
    else:
        return {"status": "error", "message": "Booking not found"}


@mcp.tool()
def clear_bookings() -> dict:
    """Clear all bookings from the database. Use between test runs to reset state."""
    df = pd.DataFrame(columns=["id", "name", "datetime", "notes", "contact", "created_at"])
    save_bookings(df)
    return {"status": "success", "message": "All bookings cleared"}


mcp_app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:mcp_app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
