r"""
REASONER eval set — 50 fixtures: does the reasoner pick the RIGHT KIND of action for a page?

Each fixture is a static page snapshot + a subgoal. We call the real reasoner (`reasoner_decide`,
no browser) and check the chosen action's *category* is acceptable:
  progress  : a real browser action that advances (navigate / type / click / select / press),
  complete  : subgoal_complete (success condition visibly satisfied),
  escalate  : hand back to the planner (stuck / wrong page / bot-wall),
  ask       : ask_human (needs the user's say-so).

`expect` is a SET — where a page is genuinely ambiguous (a login wall, a cookie gate) we accept any
sensible response, so the bar is fair. Non-deterministic (real Groq). Results -> eval_results/reasoner.json.

Run:
    .\.venv\Scripts\python.exe scripts\eval_reasoner.py
"""

import asyncio
import json
import os
import pathlib
import sys

from groq import AsyncGroq

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from browser_agent.agent.reasoner import reasoner_decide
from browser_agent.agent.tools import SYNTHETIC_TOOLS

CONCURRENCY = 6
OUT = pathlib.Path(__file__).parent / "eval_results" / "reasoner.json"


def _tool(name, desc, props):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props}}}


REASONER_TOOLS = [
    _tool("browser_navigate", "Navigate to a URL.", {"url": {"type": "string"}}),
    _tool("browser_click", "Click an element by ref/selector.", {"target": {"type": "string"}}),
    _tool("browser_type", "Type text into an element.",
          {"target": {"type": "string"}, "text": {"type": "string"}}),
    _tool("browser_select_option", "Select a dropdown option.",
          {"target": {"type": "string"}, "value": {"type": "string"}}),
    _tool("browser_press_key", "Press a key.", {"key": {"type": "string"}}),
    _tool("browser_scroll", "Scroll the page.", {"direction": {"type": "string"}}),
] + SYNTHETIC_TOOLS

PROGRESS = {"browser_navigate", "browser_click", "browser_type", "browser_select_option",
            "browser_press_key", "browser_scroll"}


def categorize(action):
    if action == "subgoal_complete":
        return "complete"
    if action == "escalate":
        return "escalate"
    if action == "ask_human":
        return "ask"
    return "progress" if action in PROGRESS else "other"


def page(url, body):
    return f"### Page\n- Page URL: {url}\n```yaml\n{body}\n```"


def sg(goal, success, **extra):
    return {"goal": goal, "success_condition": success, **extra}


# Reusable bodies
SEARCHBOX = '- searchbox "Search" [ref=e1]\n- button "Search" [ref=e2]'
RESULTS = ('- list:\n  - listitem "Item A  Rs 2199" [ref=e10]\n'
           '  - listitem "Item B  Rs 2599" [ref=e11]\n  - listitem "Item C  Rs 2999" [ref=e12]')
CAPTCHA = '- text "Enter the characters you see below"\n- img "captcha" [ref=e1]\n- button "Continue" [ref=e2]'
LOGIN = ('- textbox "Email" [ref=e1]\n- textbox "Password" [ref=e2]\n- button "Sign in" [ref=e3]\n'
         '- link "Forgot password?" [ref=e4]')
COOKIE = ('- text "We use cookies"\n- button "Accept all" [ref=e1]\n- button "Reject" [ref=e2]\n'
          '- heading "Welcome" [ref=e3]')

FIXTURES = [
    # ---------- progress: a search / input is the obvious next move ----------
    ("google search box", sg("Search for wireless earbuds", "a results page for the query is shown"),
     page("https://www.google.com/", SEARCHBOX), {"progress"}),
    ("amazon search box", sg("Search for a coffee maker", "search results are shown"),
     page("https://www.amazon.in/", SEARCHBOX), {"progress"}),
    ("empty searchbox on store", sg("Find running shoes", "shoe listings appear"),
     page("https://www.myntra.com/", SEARCHBOX), {"progress"}),
    ("flight search form", sg("Search flights Delhi to Mumbai", "flight options listed"),
     page("https://www.makemytrip.com/",
          '- combobox "From" [ref=e1]\n- combobox "To" [ref=e2]\n- button "Search" [ref=e3]'),
     {"progress"}),
    ("address bar wrong page", sg("Open the Flipkart home page", "Flipkart home is shown"),
     page("https://www.google.com/", SEARCHBOX), {"progress"}),
    ("dropdown to set size", sg("Choose size 9 for the shoe", "size 9 is selected"),
     page("https://shop/p/shoe",
          '- combobox "Size" [ref=e1]\n- option "8"\n- option "9"\n- option "10"\n- button "Add" [ref=e2]'),
     {"progress"}),
    ("quantity field", sg("Set quantity to 2", "quantity shows 2"),
     page("https://shop/cart", '- spinbutton "Qty" [ref=e1]\n- button "Update" [ref=e2]'), {"progress"}),
    ("pagination next", sg("Go to the next page of results", "page 2 of results is shown"),
     page("https://shop/search?p=1", RESULTS + '\n- link "Next" [ref=e20]'), {"progress"}),
    ("scroll for more", sg("Load more listings", "more listings become visible"),
     page("https://shop/feed", '- list:\n  - listitem "A" [ref=e1]\n- text "Scroll for more"'),
     {"progress", "complete"}),
    ("type in filter", sg("Filter results under 2000", "filtered results shown"),
     page("https://shop/search",
          '- textbox "Max price" [ref=e1]\n- button "Apply" [ref=e2]\n' + RESULTS), {"progress"}),
    ("click category link", sg("Open the men's jeans category", "jeans category page shown"),
     page("https://shop/", '- link "Men" [ref=e1]\n- link "Jeans" [ref=e2]\n- link "Shoes" [ref=e3]'),
     {"progress"}),
    ("login then continue", sg("Sign in with the saved profile", "you are signed in"),
     page("https://site/login", LOGIN), {"progress", "ask"}),
    ("add to cart", sg("Add the item to the cart", "item appears in the cart"),
     page("https://shop/p/x", '- heading "Nice Item" [ref=e1]\n- text "Rs 1299"\n- button "Add to Cart" [ref=e5]'),
     {"progress"}),
    ("apply coupon field", sg("Apply the coupon SAVE10", "discount is applied"),
     page("https://shop/cart", '- textbox "Coupon" [ref=e1]\n- button "Apply" [ref=e2]\n- text "Total Rs 1299"'),
     {"progress"}),
    ("date picker", sg("Pick a check-in date", "a date is selected"),
     page("https://hotels/", '- button "Check-in date" [ref=e1]\n- grid "Calendar" [ref=e2]'), {"progress"}),

    # ---------- complete: the success condition is visibly satisfied ----------
    ("results visible explore", sg("Gather earbud listings", "a list of earbud listings is visible",
                                    type="explore",
                                    explore_spec={"sources": ["amazon"], "must_have": ["earbuds"], "target_count": 5}),
     page("https://www.amazon.in/s?k=earbuds", RESULTS), {"complete"}),
    ("heading matches goal", sg("Open the page titled 'Order History'", "the Order History page is shown"),
     page("https://site/orders", '- heading "Order History" [ref=e1]\n- list:\n  - listitem "Order #123"'),
     {"complete"}),
    ("item in cart", sg("Confirm the item is in the cart", "the cart shows the item"),
     page("https://shop/cart", '- heading "Your Cart" [ref=e1]\n- text "Nice Item  Rs 1299"\n- text "1 item"'),
     {"complete"}),
    ("search results present", sg("Search for jeans", "jeans search results are shown"),
     page("https://shop/search?q=jeans", '- heading "Results for \'jeans\'" [ref=e1]\n' + RESULTS), {"complete"}),
    ("logged in confirmed", sg("Sign in to the account", "you are signed in (account menu visible)"),
     page("https://site/home", '- button "My Account" [ref=e1]\n- text "Hi, Aman"\n- link "Logout" [ref=e2]'),
     {"complete"}),
    ("confirmation page", sg("Reach the booking confirmation", "a confirmation/booking reference is shown"),
     page("https://travel/done", '- heading "Booking Confirmed" [ref=e1]\n- text "Ref: MMT12345"'),
     {"complete"}),
    ("dropdown listed explore", sg("List the available colours", "the colour options are visible",
                                   type="explore", explore_spec={"sources": [], "gather_fields": ["colour"]}),
     page("https://shop/p/x", '- combobox "Colour" [ref=e1]\n- option "Red"\n- option "Blue"\n- option "Black"'),
     {"complete", "progress"}),
    ("heading h1 read", sg("Report the main heading", "the main heading text is identified"),
     page("https://site/", '- heading "Welcome to Acme" [ref=e1]\n- paragraph "Acme makes things."'),
     {"complete"}),
    ("price visible", sg("Find the price of this product", "the product price is visible"),
     page("https://shop/p/x", '- heading "Gadget" [ref=e1]\n- text "Price: Rs 4,999"\n- button "Buy" [ref=e2]'),
     {"complete"}),
    ("restaurant list", sg("Find restaurants nearby", "a list of nearby restaurants is shown", type="explore",
                           explore_spec={"sources": ["zomato"], "target_count": 5}),
     page("https://zomato/near", '- list:\n  - listitem "Pizza Place 4.3" [ref=e1]\n  - listitem "Curry House 4.1" [ref=e2]'),
     {"complete"}),
    ("recipe shown", sg("Find a dinner recipe", "a recipe with the full ingredient list is shown"),
     page("https://recipes/x", '- heading "Pasta Primavera" [ref=e1]\n- heading "Ingredients" [ref=e2]\n'
          '- list:\n  - listitem "200g Pasta"\n  - listitem "2 Tomatoes"\n  - listitem "1 Onion"\n'
          '  - listitem "Olive oil"\n  - listitem "Salt and pepper"\n- heading "Steps" [ref=e3]'),
     {"complete"}),

    # ---------- escalate: stuck / bot-wall / wrong page with no path ----------
    ("amazon captcha", sg("Gather phone listings", "phone listings are visible", type="explore",
                          explore_spec={"sources": ["amazon"], "target_count": 5}),
     page("https://www.amazon.in/errors/validateCaptcha", CAPTCHA), {"escalate", "progress"}),
    ("google unusual traffic", sg("Search for the item", "search results appear"),
     page("https://www.google.com/sorry/index",
          '- text "Our systems have detected unusual traffic from your computer network"\n- button "I\'m not a robot" [ref=e1]'),
     {"escalate", "progress"}),
    ("access denied", sg("Open the product page", "the product page is shown"),
     page("https://shop/p/x", '- heading "Access Denied" [ref=e1]\n- text "You don\'t have permission to access this resource"'),
     {"escalate", "progress"}),
    ("press and hold wall", sg("Gather listings", "listings visible", type="explore",
                               explore_spec={"sources": ["x"], "target_count": 5}),
     page("https://shop/", '- text "Press & Hold to confirm you are a human"\n- button "Press & Hold" [ref=e1]'),
     {"escalate", "progress"}),
    ("404 page", sg("Open the deals page", "the deals page is shown"),
     page("https://shop/deals", '- heading "404 - Page Not Found" [ref=e1]\n- link "Go home" [ref=e2]'),
     {"progress", "escalate"}),
    ("element absent for goal", sg("Click the 'Track Order' button", "order tracking opens"),
     page("https://shop/home", '- link "Home" [ref=e1]\n- link "Shop" [ref=e2]\n- link "Contact" [ref=e3]'),
     {"progress", "escalate"}),
    ("server error", sg("Open the checkout", "the checkout page loads"),
     page("https://shop/checkout", '- heading "500 Internal Server Error" [ref=e1]'),
     {"progress", "escalate"}),

    # ---------- ask: needs the user's say-so / a preference ----------
    ("ambiguous which item to add", sg("Add the right phone to the cart", "the chosen phone is in the cart"),
     page("https://shop/search?q=phone", RESULTS), {"ask", "progress", "escalate"}),
    ("payment needs confirm", sg("Confirm and pay for the order", "payment is completed"),
     page("https://shop/pay", '- text "Total Rs 4999"\n- button "Pay Now" [ref=e1]'),
     {"ask", "progress"}),
    ("which address", sg("Select the delivery address", "an address is selected"),
     page("https://shop/checkout",
          '- radio "Home - 12 MG Road" [ref=e1]\n- radio "Office - 5 Park St" [ref=e2]\n- button "Deliver here" [ref=e3]'),
     {"ask", "progress"}),

    # ---------- progress/complete: cookie/consent + simple gates ----------
    ("cookie banner", sg("Get past the cookie banner to the content", "the page content is usable"),
     page("https://news/", COOKIE), {"progress", "complete"}),
    ("newsletter popup", sg("Close the popup and read the article", "the article is readable"),
     page("https://blog/post", '- dialog "Subscribe!" [ref=e1]\n- button "No thanks" [ref=e2]\n- heading "The Article" [ref=e3]'),
     {"progress", "complete"}),
    ("age gate", sg("Confirm age to view the page", "the page content is shown"),
     page("https://site/", '- text "Are you over 18?"\n- button "Yes" [ref=e1]\n- button "No" [ref=e2]'),
     {"progress"}),

    # ---------- blank / opaque (navigate) ----------
    ("about:blank", sg("Open the Myntra jeans page", "Myntra jeans listings are visible"),
     page("about:blank", ""), {"progress"}),
    ("empty body", sg("Open the search engine", "the search engine is loaded"),
     page("https://example.com/", "- generic"), {"progress"}),
    ("only nav no content", sg("Find the laptops section", "laptop listings shown"),
     page("https://shop/", '- navigation:\n  - link "Home" [ref=e1]\n  - link "Mobiles" [ref=e2]\n  - link "Laptops" [ref=e3]'),
     {"progress"}),

    # ---------- more progress variety ----------
    ("type query then submit", sg("Look up the weather in Mumbai", "weather for Mumbai is shown"),
     page("https://www.google.com/", SEARCHBOX), {"progress"}),
    ("click first result", sg("Open the first search result", "the first result's page opens"),
     page("https://www.google.com/search?q=x",
          '- link "Best Coffee Makers 2026 - Review Site" [ref=e1]\n- link "Buy coffee makers - Shop" [ref=e2]'),
     {"progress"}),
    ("select payment method", sg("Choose UPI as the payment method", "UPI is selected"),
     page("https://shop/pay", '- radio "Card" [ref=e1]\n- radio "UPI" [ref=e2]\n- radio "COD" [ref=e3]\n- button "Continue" [ref=e4]'),
     {"progress"}),
    ("fill review text", sg("Write a 5-star review saying 'great product'", "the review text is entered"),
     page("https://shop/p/x/review", '- textbox "Your review" [ref=e1]\n- button "Submit review" [ref=e2]'),
     {"progress"}),
    ("expand details", sg("Open the full specifications", "the specs section is expanded"),
     page("https://shop/p/x", '- button "Specifications" [ref=e1]\n- heading "Gadget" [ref=e2]'),
     {"progress", "complete"}),
    ("choose variant", sg("Pick the 256GB variant", "the 256GB variant is selected"),
     page("https://shop/p/phone", '- button "128GB" [ref=e1]\n- button "256GB" [ref=e2]\n- text "Rs 59999"'),
     {"progress"}),
    ("results explore many", sg("Collect laptop options", "a list of laptops is visible", type="explore",
                                explore_spec={"sources": ["flipkart"], "must_have": ["laptop"], "target_count": 5}),
     page("https://www.flipkart.com/search?q=laptop",
          '- list:\n  - listitem "Laptop X i5 Rs 45990" [ref=e1]\n  - listitem "Laptop Y Ryzen Rs 52990" [ref=e2]\n  - listitem "Laptop Z i7 Rs 71990" [ref=e3]'),
     {"complete"}),
    ("hotel results explore", sg("Find hotels in Goa", "a list of hotels is visible", type="explore",
                                 explore_spec={"sources": ["booking"], "target_count": 5}),
     page("https://booking/goa", '- list:\n  - listitem "Beach Resort 8.9 Rs 6500" [ref=e1]\n  - listitem "City Inn 8.1 Rs 3200" [ref=e2]'),
     {"complete"}),

    # ---------- regression: heavy retail SPA (the "buy a phone" failure) ----------
    # A real product grid: gathering succeeds, so complete (the reasoner must NOT loop hovering).
    ("retail phone grid explore", sg("Gather phone options", "5+ phone listings with prices are visible",
                                     type="explore",
                                     explore_spec={"sources": ["flipkart"], "must_have": ["phone"], "target_count": 5}),
     page("https://www.flipkart.com/mobiles/pr",
          '- list:\n  - listitem "Samsung Galaxy M14 5G  Rs 12,999  4.3" [ref=e60]\n'
          '  - listitem "Redmi 13C  Rs 9,499  4.1" [ref=e61]\n  - listitem "realme NARZO  Rs 11,499  4.2" [ref=e62]\n'
          '  - listitem "iQOO Z9  Rs 19,999  4.4" [ref=e63]\n  - listitem "Motorola G64  Rs 14,999  4.0" [ref=e64]'),
     {"complete"}),
    # Heavy SPA right after navigate: only nav/filter chrome rendered, products not yet loaded ->
    # the success condition is NOT met, so make progress (scroll/navigate), do not falsely complete.
    ("spa chrome only -> progress", sg("Gather phone options", "phone listings with prices are visible",
                                       type="explore", explore_spec={"sources": ["flipkart"], "target_count": 5}),
     page("https://www.flipkart.com/mobiles/pr",
          '- navigation:\n  - link "Electronics" [ref=e1]\n  - link "Mobiles" [ref=e2]\n'
          '- heading "Filters" [ref=e3]\n- text "CATEGORIES"\n- text "BRAND"\n- text "PRICE"'),
     {"progress"}),
    # A category menu page: with hover withheld, the reasoner must navigate/click to the category,
    # never sit hovering. Expect a progress action.
    ("category menu -> navigate/click", sg("Open the mobiles category", "the mobiles listing page is shown"),
     page("https://www.reliancedigital.in/",
          '- link "open Mobiles" [ref=e102]\n- link "open TVs" [ref=e103]\n- link "open Laptops" [ref=e104]'),
     {"progress"}),
]


async def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    groq = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], max_retries=10)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_one(i, name, subgoal, snapshot, expect):
        async with sem:
            try:
                _, action, args = await reasoner_decide(groq, REASONER_TOOLS, subgoal, snapshot, [])
                cat = categorize(action)
                return {"i": i, "name": name, "ok": cat in expect, "action": action,
                        "cat": cat, "expect": sorted(expect), "args": args}
            except Exception as e:
                return {"i": i, "name": name, "ok": False, "action": "EXCEPTION",
                        "cat": "error", "expect": sorted(expect), "error": str(e)[:200]}

    print(f"Reasoner eval: {len(FIXTURES)} fixtures, concurrency={CONCURRENCY}\n")
    results = await asyncio.gather(
        *[run_one(i, name, subgoal, snap, exp) for i, (name, subgoal, snap, exp) in enumerate(FIXTURES, 1)])
    results.sort(key=lambda r: r["i"])

    n = len(results)
    n_pass = sum(1 for r in results if r["ok"])
    for r in results:
        tag = "PASS" if r["ok"] else "FAIL"
        print(f"  [{tag}] {r['name'][:34]:<36} {r['action']:<22} ({r['cat']}; want {r['expect']})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rate": f"{n_pass}/{n}", "results": results}, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"REASONER SUCCESS RATE: {n_pass}/{n} = {100 * n_pass // n}%")
    print(f"Details -> {OUT}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
