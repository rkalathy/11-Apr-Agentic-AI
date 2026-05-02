import os
import logging
from dotenv import load_dotenv

load_dotenv()

from agno.agent import Agent
from agno.models.google import Gemini
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ---------------------------------------------------------------------------
# Mock order database
# ---------------------------------------------------------------------------
ORDERS: dict = {
    "ORD001": {"customer": "Alice Johnson",  "amount": 150.00, "item": "Laptop Bag",           "status": "delivered"},
    "ORD002": {"customer": "Bob Smith",      "amount":  89.99, "item": "Wireless Mouse",        "status": "delivered"},
    "ORD003": {"customer": "Charlie Brown",  "amount": 299.00, "item": "Mechanical Keyboard",   "status": "shipped"},
    "ORD004": {"customer": "Diana Prince",   "amount":  49.99, "item": "USB-C Hub",             "status": "delivered"},
}

# Keyed by refund_key, holds request metadata and approval status
pending_refunds: dict = {}

# ---------------------------------------------------------------------------
# Agent tools (plain sync functions — agno discovers the signature + docstring)
# ---------------------------------------------------------------------------

def lookup_order(order_id: str) -> str:
    """
    Look up an order by its order ID and return its details.

    Args:
        order_id: The order ID (e.g. ORD001).

    Returns:
        A string with order details, or an error message if not found.
    """
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"Order '{order_id}' not found in the system."
    o = order
    return (
        f"Order {order_id.upper()}: "
        f"Customer={o['customer']}, "
        f"Amount=${o['amount']:.2f}, "
        f"Item={o['item']}, "
        f"Status={o['status']}"
    )


def request_refund(order_id: str, reason: str) -> str:
    """
    Queue a refund request for human approval.  A Telegram confirmation message
    with Approve / Reject buttons will be sent automatically.

    Args:
        order_id: The order ID to refund (e.g. ORD001).
        reason:   The reason the customer is requesting a refund.

    Returns:
        Confirmation that the request has been queued.
    """
    oid = order_id.upper()
    order = ORDERS.get(oid)
    if not order:
        return f"Cannot queue refund: Order '{order_id}' not found."
    if order["status"] == "refunded":
        return f"Order '{oid}' has already been refunded."

    refund_key = f"{oid}_{len(pending_refunds)}"
    pending_refunds[refund_key] = {
        "order_id": oid,
        "reason": reason,
        "order": order.copy(),
        "status": "pending",
    }
    return (
        f"Refund request queued for Order {oid} (${order['amount']:.2f}). "
        f"Key: {refund_key}. Awaiting human approval via Telegram."
    )


# ---------------------------------------------------------------------------
# Telegram confirmation helpers
# ---------------------------------------------------------------------------

async def _send_confirmation(app: Application, refund_key: str, chat_id: str | int):
    """Send an inline-keyboard confirmation card to the given chat."""
    info = pending_refunds[refund_key]
    order = info["order"]
    oid = info["order_id"]

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve Refund", callback_data=f"approve_{refund_key}"),
            InlineKeyboardButton("❌ Reject Refund",  callback_data=f"reject_{refund_key}"),
        ]
    ])

    text = (
        f"🔔 *Refund Approval Required*\n\n"
        f"📦 Order ID: `{oid}`\n"
        f"👤 Customer: {order['customer']}\n"
        f"💰 Amount: `${order['amount']:.2f}`\n"
        f"🛍️ Item: {order['item']}\n"
        f"📋 Order Status: {order['status']}\n"
        f"📝 Reason: _{info['reason']}_\n\n"
        f"Please approve or reject this refund request:"
    )

    await app.bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode="Markdown",
        reply_markup=keyboard,
    )


# ---------------------------------------------------------------------------
# Telegram message handler — invokes the Agno agent
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    app: Application = context.application

    await update.message.reply_text("🤖 Processing your request…")

    # Snapshot pending_refunds before the agent runs so we know which keys are new
    keys_before = set(pending_refunds.keys())

    agent = Agent(
        model=Gemini(id="gemini-2.5-flash"),
        tools=[lookup_order, request_refund],
        instructions=(
            "You are a customer-support agent that handles refund requests. "
            "When asked about a refund: "
            "1. Call lookup_order to verify the order exists and is eligible. "
            "2. Call request_refund with the order ID and the customer's reason. "
            "3. Inform the customer that a human operator will review and approve/reject "
            "   the request shortly via Telegram. "
            "Be concise and professional."
        ),
        markdown=True,
    )

    try:
        response = agent.run(user_text)
        reply = response.content or "✅ Request processed."
    except Exception as exc:
        logger.exception("Agent error")
        reply = f"⚠️ An error occurred: {exc}"

    await update.message.reply_text(reply, parse_mode="Markdown")

    # Send Telegram confirmation cards for every new pending refund
    new_keys = set(pending_refunds.keys()) - keys_before
    for key in new_keys:
        await _send_confirmation(app, key, chat_id)


# ---------------------------------------------------------------------------
# Callback handler — handles Approve / Reject button presses
# ---------------------------------------------------------------------------

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data: str = query.data

    if data.startswith("approve_"):
        refund_key = data[len("approve_"):]
        info = pending_refunds.get(refund_key)

        if not info or info["status"] != "pending":
            await query.edit_message_text("⚠️ This refund request is no longer active.")
            return

        oid = info["order_id"]
        ORDERS[oid]["status"] = "refunded"
        info["status"] = "approved"

        await query.edit_message_text(
            f"✅ *Refund Approved & Processed*\n\n"
            f"📦 Order: `{oid}`\n"
            f"👤 Customer: {info['order']['customer']}\n"
            f"💰 Refunded: `${info['order']['amount']:.2f}`\n"
            f"📝 Reason: _{info['reason']}_\n\n"
            f"The customer will receive their refund within 3–5 business days.",
            parse_mode="Markdown",
        )
        logger.info("Refund APPROVED for order %s", oid)

    elif data.startswith("reject_"):
        refund_key = data[len("reject_"):]
        info = pending_refunds.get(refund_key)

        if not info or info["status"] != "pending":
            await query.edit_message_text("⚠️ This refund request is no longer active.")
            return

        info["status"] = "rejected"
        oid = info["order_id"]

        await query.edit_message_text(
            f"❌ *Refund Rejected*\n\n"
            f"📦 Order: `{oid}`\n"
            f"👤 Customer: {info['order']['customer']}\n"
            f"💰 Amount: `${info['order']['amount']:.2f}`\n"
            f"📝 Reason for request: _{info['reason']}_\n\n"
            f"The refund has been rejected. The customer will be notified.",
            parse_mode="Markdown",
        )
        logger.info("Refund REJECTED for order %s", oid)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Add it to your .env file and restart."
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("Refund Agent bot is running. Message your Telegram bot to start.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
