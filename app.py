"""
MovieMind - AI-Powered Movie Recommendation System

A content-based movie recommendation engine using cosine similarity
on movie features (genres, keywords, cast, director, overview).

NOTE: Only the presentation layer (CSS / layout / markup) has been redesigned.
All backend, ML, data-loading, caching and TMDB logic is unchanged.
"""

import streamlit as st
from src import (
    load_movies_data,
    load_similarity_matrix,
    MovieRecommender,
    TMDBClient,
    PAGE_CONFIG,
)

# ============================================================================
# PAGE CONFIG & STYLING  (UI only)
# ============================================================================

st.set_page_config(**PAGE_CONFIG)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{
    --mm-bg:#08090f;
    --mm-bg-2:#0b0e1a;
    --mm-surface:rgba(255,255,255,.035);
    --mm-surface-2:rgba(255,255,255,.06);
    --mm-border:rgba(255,255,255,.09);
    --mm-border-strong:rgba(255,255,255,.16);
    --mm-violet:#8b5cf6;
    --mm-violet-soft:#a78bfa;
    --mm-blue:#3b82f6;
    --mm-text:#f4f5f8;
    --mm-muted:#9aa1b1;
    --mm-radius:18px;
    --mm-shadow:0 18px 50px -20px rgba(0,0,0,.85);
}

html, body, .stApp{
    background:
        radial-gradient(1100px 620px at 12% -8%, rgba(139,92,246,.16), transparent 60%),
        radial-gradient(900px 560px at 88% 0%, rgba(59,130,246,.12), transparent 60%),
        linear-gradient(180deg, var(--mm-bg-2) 0%, var(--mm-bg) 55%, #06070c 100%);
    color: var(--mm-text);
    font-family:'Inter', system-ui, sans-serif;
}
.stApp > header{ background:transparent; }
.block-container{ padding-top:1.2rem; max-width:1280px; }

h1,h2,h3,h4{ font-family:'Sora', system-ui, sans-serif; color:var(--mm-text); letter-spacing:-.02em; border:none !important; }
h2{ font-size:1.6rem; padding-bottom:0 !important; }
h3{ color:var(--mm-text); font-size:1.2rem; }
p, li, span, label{ color:var(--mm-muted); }
strong{ color:var(--mm-text); }
hr, .stDivider{ border-color:var(--mm-border) !important; }

/* ---------- Top navigation ---------- */
.mm-nav{
    display:flex; align-items:center; justify-content:space-between; gap:24px;
    padding:14px 22px; margin-bottom:22px;
    background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
    border:1px solid var(--mm-border); border-radius:16px;
    backdrop-filter:blur(14px);
}
.mm-brand{ display:flex; align-items:center; gap:12px; }
.mm-mark{
    width:38px; height:38px; border-radius:12px;
    background:linear-gradient(135deg, var(--mm-violet), var(--mm-blue));
    display:flex; align-items:center; justify-content:center; font-size:19px;
    box-shadow:0 8px 22px -8px rgba(139,92,246,.9);
}
.mm-brand-name{ font-family:'Sora',sans-serif; font-weight:800; font-size:1.05rem; color:var(--mm-text); line-height:1.1; }
.mm-brand-sub{ font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--mm-muted); }
.mm-navlinks{ display:flex; gap:8px; flex-wrap:wrap; }
.mm-navlink{
    font-size:.82rem; color:var(--mm-muted); padding:7px 14px; border-radius:999px;
    border:1px solid transparent; transition:all .25s ease;
}
.mm-navlink.active{ color:var(--mm-text); background:var(--mm-surface-2); border-color:var(--mm-border-strong); }

/* ---------- Hero ---------- */
.mm-hero{
    position:relative; overflow:hidden; text-align:center;
    padding:64px 28px 58px; margin-bottom:26px;
    border:1px solid var(--mm-border); border-radius:26px;
    background:
        radial-gradient(700px 320px at 50% 0%, rgba(139,92,246,.22), transparent 70%),
        linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
    box-shadow:var(--mm-shadow);
}
.mm-hero::after{
    content:""; position:absolute; inset:0; pointer-events:none;
    background:linear-gradient(90deg, transparent, rgba(59,130,246,.07), transparent);
}
.mm-pill{
    display:inline-flex; align-items:center; gap:8px; font-size:.75rem; letter-spacing:.1em;
    text-transform:uppercase; color:var(--mm-violet-soft);
    border:1px solid rgba(139,92,246,.35); background:rgba(139,92,246,.12);
    padding:6px 14px; border-radius:999px; margin-bottom:20px;
}
.mm-hero h1{
    font-size:clamp(2.1rem, 4.6vw, 3.5rem); font-weight:800; line-height:1.08; margin:0 auto;
    max-width:16ch;
    background:linear-gradient(180deg,#ffffff 30%, #b9bfd0 100%);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.mm-hero p{ margin:16px auto 0; max-width:52ch; font-size:1.03rem; color:var(--mm-muted); }

/* ---------- Panels ---------- */
.mm-panel{
    background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
    border:1px solid var(--mm-border); border-radius:var(--mm-radius);
    padding:22px 24px; box-shadow:var(--mm-shadow); margin-bottom:6px;
}
.mm-eyebrow{ font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:var(--mm-violet-soft); margin-bottom:6px; }

/* ---------- Selectbox as SaaS search field ---------- */
div[data-testid="stSelectbox"] label{ font-size:.75rem; letter-spacing:.14em; text-transform:uppercase; color:var(--mm-muted); }
div[data-baseweb="select"] > div{
    background:rgba(255,255,255,.05) !important;
    border:1px solid var(--mm-border-strong) !important;
    border-radius:14px !important; min-height:52px; color:var(--mm-text) !important;
    transition:border-color .25s ease, box-shadow .25s ease;
}
div[data-baseweb="select"] > div:hover{ border-color:rgba(139,92,246,.55) !important; }
div[data-baseweb="select"] > div:focus-within{
    border-color:var(--mm-violet) !important; box-shadow:0 0 0 4px rgba(139,92,246,.18) !important;
}
div[data-baseweb="select"] input, div[data-baseweb="select"] div{ color:var(--mm-text) !important; }
div[data-baseweb="popover"] div[role="listbox"]{
    background:#0d1020 !important; border:1px solid var(--mm-border-strong) !important; border-radius:14px !important;
}
div[data-baseweb="popover"] li:hover{ background:rgba(139,92,246,.18) !important; }

/* ---------- Buttons ---------- */
.stButton > button{
    background:linear-gradient(135deg, var(--mm-violet) 0%, var(--mm-blue) 100%);
    color:#fff !important; border:none; min-height:52px; padding:12px 26px;
    border-radius:14px; font-weight:600; font-family:'Sora',sans-serif; letter-spacing:.01em;
    box-shadow:0 14px 30px -14px rgba(139,92,246,.9); transition:all .25s ease; width:100%;
}
.stButton > button:hover{ transform:translateY(-2px); filter:brightness(1.08); box-shadow:0 20px 40px -14px rgba(99,102,241,.95); }
.stButton > button:active{ transform:translateY(0); }

/* ---------- Movie cards ---------- */
.mm-card{
    background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
    border:1px solid var(--mm-border); border-radius:18px; overflow:hidden;
    box-shadow:var(--mm-shadow); transition:transform .3s ease, border-color .3s ease, box-shadow .3s ease;
    height:100%; display:flex; flex-direction:column;
}
.mm-card:hover{ transform:translateY(-6px); border-color:rgba(139,92,246,.5); box-shadow:0 26px 55px -22px rgba(139,92,246,.6); }
.mm-poster{ position:relative; aspect-ratio:2/3; overflow:hidden; background:#11131f; }
.mm-poster img{ width:100%; height:100%; object-fit:cover; display:block; transition:transform .5s ease; }
.mm-card:hover .mm-poster img{ transform:scale(1.07); }
.mm-poster::after{
    content:""; position:absolute; inset:0;
    background:linear-gradient(180deg, transparent 55%, rgba(8,9,15,.85) 100%);
}
.mm-card-body{ padding:14px 14px 16px; display:flex; flex-direction:column; gap:10px; flex:1; }
.mm-card-title{
    font-family:'Sora',sans-serif; font-size:.92rem; font-weight:600; color:var(--mm-text);
    line-height:1.3; min-height:2.6em;
}
.mm-badge{
    display:inline-flex; align-items:center; gap:7px; align-self:flex-start;
    font-size:.75rem; font-weight:600; color:var(--mm-violet-soft);
    background:rgba(139,92,246,.13); border:1px solid rgba(139,92,246,.32);
    padding:5px 11px; border-radius:999px;
}
.mm-dot{ width:7px; height:7px; border-radius:50%; background:linear-gradient(135deg,var(--mm-violet),var(--mm-blue)); box-shadow:0 0 8px rgba(139,92,246,.9); }

/* ---------- Selected movie / section headers ---------- */
.mm-section-label{ font-size:.72rem; letter-spacing:.2em; text-transform:uppercase; color:var(--mm-muted); }
.mm-selected{
    display:flex; align-items:center; gap:14px; flex-wrap:wrap;
    padding:18px 22px; margin:8px 0 22px; border-radius:16px;
    border:1px solid var(--mm-border); background:linear-gradient(90deg, rgba(139,92,246,.14), rgba(59,130,246,.06));
}
.mm-selected-title{ font-family:'Sora',sans-serif; font-size:1.25rem; font-weight:700; color:var(--mm-text); }

/* ---------- Empty state ---------- */
.mm-empty{
    text-align:center; padding:56px 24px; border-radius:20px; margin-top:8px;
    border:1px dashed var(--mm-border-strong); background:rgba(255,255,255,.02);
}
.mm-empty-icon{ font-size:2.6rem; }
.mm-empty h3{ margin:14px 0 6px; font-size:1.25rem; }
.mm-empty p{ margin:0; }

/* ---------- Steps ---------- */
.mm-step{
    position:relative; padding:20px 22px; border-radius:16px; margin-bottom:14px;
    border:1px solid var(--mm-border);
    background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
    transition:transform .25s ease, border-color .25s ease;
}
.mm-step:hover{ transform:translateX(6px); border-color:rgba(139,92,246,.45); }
.mm-step-num{
    font-family:'Sora',sans-serif; font-size:.8rem; font-weight:700; letter-spacing:.12em;
    color:transparent; background:linear-gradient(135deg,var(--mm-violet),var(--mm-blue));
    -webkit-background-clip:text; background-clip:text;
}
.mm-step h4{ margin:4px 0 8px; font-size:1.05rem; }
.mm-step p{ margin:0; font-size:.9rem; }
.mm-connector{ height:20px; width:1px; margin:0 auto 14px; background:linear-gradient(180deg, rgba(139,92,246,.6), transparent); }

/* ---------- Feature chips ---------- */
.mm-chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.mm-chip{
    font-size:.78rem; color:var(--mm-text); padding:6px 12px; border-radius:999px;
    border:1px solid var(--mm-border-strong); background:rgba(255,255,255,.045);
}

/* ---------- Metrics ---------- */
div[data-testid="stMetric"]{
    background:linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
    border:1px solid var(--mm-border); border-radius:16px; padding:20px 22px;
    box-shadow:var(--mm-shadow);
}
div[data-testid="stMetricLabel"] p{ font-size:.72rem !important; letter-spacing:.16em; text-transform:uppercase; color:var(--mm-muted) !important; }
div[data-testid="stMetricValue"]{ font-family:'Sora',sans-serif; color:var(--mm-text) !important; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"]{
    gap:6px; background:rgba(255,255,255,.03); border:1px solid var(--mm-border);
    padding:6px; border-radius:14px;
}
.stTabs [data-baseweb="tab-list"] button{
    color:var(--mm-muted); border-radius:10px; padding:8px 18px; font-weight:500;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"]{
    color:#fff; background:linear-gradient(135deg, rgba(139,92,246,.9), rgba(59,130,246,.85));
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }

/* ---------- Alerts ---------- */
div[data-testid="stAlert"]{ border-radius:14px; border:1px solid var(--mm-border-strong); }

/* ---------- Responsive grid ---------- */
@media (max-width:1100px){
    div[data-testid="stHorizontalBlock"]:has(.mm-card){ flex-wrap:wrap; }
    div[data-testid="stHorizontalBlock"]:has(.mm-card) > div[data-testid="stColumn"]{ flex:0 0 31%; min-width:31%; }
}
@media (max-width:720px){
    div[data-testid="stHorizontalBlock"]:has(.mm-card) > div[data-testid="stColumn"]{ flex:0 0 47%; min-width:47%; }
    .mm-nav{ flex-direction:column; align-items:flex-start; }
    .mm-hero{ padding:44px 18px; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE LOADING OF EXPENSIVE RESOURCES  (unchanged)
# ============================================================================

@st.cache_resource
def load_recommendation_system():
    """Load movies data, similarity matrix, and initialize recommender engine."""
    try:
        movies = load_movies_data()
        similarity = load_similarity_matrix()
        return MovieRecommender(movies, similarity)
    except Exception as e:
        st.error(f"❌ Failed to load recommendation system: {e}")
        return None


@st.cache_resource
def get_tmdb_client():
    """Initialize TMDB API client."""
    return TMDBClient()


# ============================================================================
# UI HELPERS (presentation only)
# ============================================================================

def render_nav(active: str):
    links = ["Home", "Discover", "How It Works", "About"]
    items = "".join(
        f'<span class="mm-navlink{" active" if l == active else ""}">{l}</span>'
        for l in links
    )
    st.markdown(f"""
    <div class="mm-nav">
        <div class="mm-brand">
            <div class="mm-mark">🎬</div>
            <div>
                <div class="mm-brand-name">MovieMind</div>
                <div class="mm-brand-sub">AI Movie Discovery</div>
            </div>
        </div>
        <div class="mm-navlinks">{items}</div>
    </div>
    """, unsafe_allow_html=True)


def render_step(num: str, title: str, body: str, last: bool = False):
    st.markdown(f"""
    <div class="mm-step">
        <div class="mm-step-num">{num}</div>
        <h4>{title}</h4>
        <p>{body}</p>
    </div>
    {"" if last else '<div class="mm-connector"></div>'}
    """, unsafe_allow_html=True)


# ============================================================================
# APP LOGIC
# ============================================================================

def main():
    """Main application logic."""

    # Initialize recommender
    recommender = load_recommendation_system()
    if recommender is None:
        return

    tmdb_client = get_tmdb_client()

    # Create tabs for navigation
    tab1, tab2, tab3 = st.tabs(["🎬 Discover", "❓ How It Works", "ℹ️ About"])

    with tab1:
        render_nav("Discover")

        if not tmdb_client.api_key:
            st.info("💡 **Tip:** TMDB API key is not configured. Posters will display placeholders. Add `TMDB_API_KEY = \"...\"` in your Streamlit Cloud **Secrets** to show live movie posters.")

        # Hero Section
        st.markdown("""
        <div class="mm-hero">
            <div class="mm-pill">✦ AI-Powered Discovery</div>
            <h1>Discover your next favorite movie.</h1>
            <p>AI-powered recommendations based on the movies you already love.</p>
        </div>
        """, unsafe_allow_html=True)

        # Selection and Recommendation Section
        st.markdown("""
        <div class="mm-eyebrow">Start here</div>
        <h2>Pick a movie you love</h2>
        """, unsafe_allow_html=True)
        st.markdown(
            "Select a movie and our content-based model will surface the five closest matches."
        )

        col_select, col_button = st.columns([3, 1])

        with col_select:
            selected_movie = st.selectbox(
                "Search the catalogue",
                options=recommender.get_available_movies(),
                key="movie_selector"
            )

        with col_button:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            recommend_button = st.button("✨ Get Recommendations", use_container_width=True)

        if recommend_button:
            try:
                with st.spinner("Analysing similarity across the catalogue…"):
                    # Get recommendations (unchanged backend call)
                    titles, movie_ids, scores = recommender.recommend(selected_movie, num_recommendations=5)
                    posters = [tmdb_client.fetch_poster(movie_id) for movie_id in movie_ids]

                st.markdown(f"""
                <div class="mm-selected">
                    <span class="mm-section-label">Because you liked</span>
                    <span class="mm-selected-title">{selected_movie}</span>
                </div>
                <div class="mm-eyebrow">Recommended for you</div>
                <h2>Top 5 matches</h2>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                # Create 5-column layout for movie cards
                cols = st.columns(5)

                for idx, (col, title, poster_url, score) in enumerate(zip(cols, titles, posters, scores)):
                    with col:
                        st.markdown(f"""
                        <div class="mm-card">
                            <div class="mm-poster">
                                <img src="{poster_url}" alt="{title} poster" loading="lazy" />
                            </div>
                            <div class="mm-card-body">
                                <div class="mm-card-title">{title}</div>
                                <div class="mm-badge"><span class="mm-dot"></span>{score*100:.1f}% Match</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
                st.markdown("""
                <div class="mm-panel">
                    <div class="mm-eyebrow">More details</div>
                    <p style="margin:0">Match percentages come from the pre-computed cosine similarity
                    matrix. Pick another title above to explore a different neighbourhood of the catalogue.</p>
                </div>
                """, unsafe_allow_html=True)

            except ValueError as e:
                st.warning(f"⚠️ {e}")
            except Exception as e:
                st.error(f"❌ Error getting recommendations: {e}")
        else:
            st.markdown("""
            <div class="mm-empty">
                <div class="mm-empty-icon">🎬</div>
                <h3>Ready to discover something new?</h3>
                <p>Select a movie above and let MovieMind find your next favorites.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        render_nav("How It Works")

        st.markdown("""
        <div class="mm-eyebrow">Under the hood</div>
        <h2>How the recommendation engine works</h2>
        <p style="max-width:62ch">MovieMind uses <strong>content-based filtering</strong> with
        cosine similarity over movie features. Here is the exact pipeline behind every result.</p>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        render_step("01", "🎞️ Movie Features",
                    "For each movie we extract genres, plot keywords, the top 4 cast members, "
                    "the director and the overview text.")
        render_step("02", "🧹 Text Processing",
                    "Features are lowercased, multi-word terms are collapsed, and Porter Stemming "
                    "reduces every word to its root form.")
        render_step("03", "🔢 Vectorization",
                    "<strong>CountVectorizer</strong> maps the text into a 5000-dimensional space where "
                    "each dimension is a unique word and the value is its frequency.")
        render_step("04", "📐 Similarity",
                    "<strong>Cosine similarity</strong> measures the angle between two feature vectors, "
                    "from 0 (unrelated) to 1 (identical). It is pre-computed once and loaded instantly.")
        render_step("05", "⭐ Recommendation",
                    "Scores for the selected movie are sorted highest-first and the top 5 titles "
                    "(excluding the movie itself) are returned.", last=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="mm-panel">
            <div class="mm-eyebrow">Why content-based?</div>
            <div class="mm-chips">
                <span class="mm-chip">No cold-start problem</span>
                <span class="mm-chip">Transparent features</span>
                <span class="mm-chip">Instant, pre-computed</span>
                <span class="mm-chip">Explainable results</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dataset info
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("### Dataset")
        total_movies = len(recommender.movies)
        st.metric("Total Movies in Database", f"{total_movies:,}")
        st.caption("Source: TMDB 5000 Movies Dataset")

    with tab3:
        render_nav("About")

        st.markdown("""
        <div class="mm-eyebrow">About</div>
        <h2>MovieMind</h2>
        <p style="max-width:62ch">An AI-powered movie recommendation system built as a portfolio
        project to demonstrate machine learning and software engineering principles.</p>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="mm-panel">
                <div class="mm-eyebrow">Technology stack</div>
                <div class="mm-chips">
                    <span class="mm-chip">Python</span>
                    <span class="mm-chip">Streamlit</span>
                    <span class="mm-chip">Pandas</span>
                    <span class="mm-chip">NumPy</span>
                    <span class="mm-chip">Scikit-learn</span>
                    <span class="mm-chip">TMDB API</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="mm-panel">
                <div class="mm-eyebrow">Algorithm</div>
                <p style="margin:0 0 6px"><strong>Type:</strong> Content-Based Filtering</p>
                <p style="margin:0 0 6px"><strong>Similarity:</strong> Cosine Similarity</p>
                <p style="margin:0"><strong>Features:</strong> Genres, Keywords, Cast, Director, Overview</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="mm-panel">
                <div class="mm-eyebrow">Features</div>
                <p style="margin:0 0 6px">✅ Content-based recommendation engine</p>
                <p style="margin:0 0 6px">✅ Fast, pre-computed similarity matrix</p>
                <p style="margin:0 0 6px">✅ Movie poster integration</p>
                <p style="margin:0 0 6px">✅ Clean, modern UI</p>
                <p style="margin:0 0 6px">✅ Error handling and fallback images</p>
                <p style="margin:0">✅ Comprehensive documentation</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="mm-panel">
                <div class="mm-eyebrow">Limitations</div>
                <p style="margin:0 0 6px">Based only on movie content, not user ratings or behaviour</p>
                <p style="margin:0 0 6px">Limited by TMDB API availability</p>
                <p style="margin:0">Static movie database with no real-time updates</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="mm-panel">
            <div class="mm-eyebrow">Future improvements</div>
            <div class="mm-chips">
                <span class="mm-chip">Collaborative filtering</span>
                <span class="mm-chip">Hybrid approach</span>
                <span class="mm-chip">User preferences &amp; ratings</span>
                <span class="mm-chip">Embeddings &amp; transformers</span>
                <span class="mm-chip">Real-time dataset updates</span>
                <span class="mm-chip">Accounts &amp; personalization</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="mm-panel">
            <div class="mm-eyebrow">Contact &amp; links</div>
            <p style="margin:0"><a href="#">GitHub Repository</a> · <a href="#">Portfolio</a> · <a href="#">TMDB API</a></p>
        </div>
        """, unsafe_allow_html=True)

        # Show some stats
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Movies Available", f"{len(recommender.movies):,}")
        with col2:
            st.metric("Recommendations Per Query", "5")
        with col3:
            st.metric("Algorithm Type", "Content-Based")

        st.caption("Built with ❤️ using Streamlit and Machine Learning")


if __name__ == "__main__":
    main()
