//! A board's card-by-card analysis as a [`DoubleDummyPlay`], the value
//! `bridge-encodings` writes as the PBN `[DoubleDummyPlay]` section.
//!
//! Every card played is a node: [`node_alternatives`] solves each legal card
//! there, and its `cost` (tricks given away against the side's best choice)
//! is the section's value. The section's specification is bridge-encodings'
//! `docs/double-dummy-play.md`.

use bridge_types::{
    Board, Card, CardCosts, Contract, Direction, DoubleDummyPlay, PlaySequence, Rank, Strain, Suit,
    TrickCosts,
};

use crate::analyse_play::{node_alternatives, parse_card, PlayError, PlayInput};
use crate::convert::direction_to_seat;
use crate::{Hands, CLUB, DIAMOND, HEART, NOTRUMP, SPADE};

/// Why a board's play could not be analysed.
#[derive(Debug, Clone, PartialEq)]
pub enum DdPlayError {
    /// No `[Play]`, or nothing in it.
    NoPlay,
    /// No contract and declarer to analyse the play against, or a passed-out
    /// board.
    NoContract,
    /// The deal is missing cards.
    IncompleteDeal,
    /// The opening lead is not from declarer's left.
    WrongLeader,
    /// The record is not legal play: a card not held, or a revoke.
    Play(PlayError),
    /// The caller asked to stop.
    Cancelled,
}

impl std::fmt::Display for DdPlayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DdPlayError::NoPlay => write!(f, "no play on record"),
            DdPlayError::NoContract => write!(f, "no contract and declarer"),
            DdPlayError::IncompleteDeal => write!(f, "the deal is incomplete"),
            DdPlayError::WrongLeader => write!(f, "the opening lead is not from declarer's left"),
            DdPlayError::Play(e) => write!(f, "{e}"),
            DdPlayError::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for DdPlayError {}

/// Analyse every card of `board`'s play.
///
/// The play is analysed up to its first card not on record (`-`); turns from
/// there on are left out, as the section's `-` says. `progress` is called
/// before each card with `(done, total)` and stops the analysis by returning
/// `false`, for a caller that runs this off the main thread.
pub fn analyse_board_play(
    board: &Board,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Result<DoubleDummyPlay, DdPlayError> {
    let play = board.play.as_ref().ok_or(DdPlayError::NoPlay)?;
    let contract = board
        .contract
        .as_deref()
        .and_then(Contract::parse)
        .ok_or(DdPlayError::NoContract)?;
    let declarer = board.declarer.ok_or(DdPlayError::NoContract)?;
    if play.opening_leader != declarer.next() {
        return Err(DdPlayError::WrongLeader);
    }
    let hands =
        Hands::from_pbn(&board.deal.to_pbn(Direction::North)).ok_or(DdPlayError::IncompleteDeal)?;
    if !hands.is_complete() {
        return Err(DdPlayError::IncompleteDeal);
    }

    let played = played_in_order(play);
    if played.is_empty() {
        return Err(DdPlayError::NoPlay);
    }
    let input = PlayInput {
        hands,
        trump: trump_of(contract.strain),
        declarer: direction_to_seat(declarer),
        leader: direction_to_seat(play.opening_leader),
        plays: played.iter().map(|&c| solver_card(c)).collect(),
    };

    let mut tricks: Vec<TrickCosts> = play
        .tricks
        .iter()
        .map(|t| TrickCosts {
            leader: t.leader,
            turns: [None, None, None, None],
        })
        .collect();
    for node in 0..input.plays.len() {
        if !progress(node, input.plays.len()) {
            return Err(DdPlayError::Cancelled);
        }
        let analysis = node_alternatives(&input, node).map_err(DdPlayError::Play)?;
        let mut cards: Vec<(Card, Option<u8>)> = analysis
            .alternatives
            .iter()
            .filter_map(|a| Some((card_from_name(&a.card)?, Some(a.cost))))
            .collect();
        // The section's order: spades down to clubs, high to low
        cards.sort_by_key(|(c, _)| (suit_order(c.suit), std::cmp::Reverse(c.rank)));
        tricks[node / 4].turns[node % 4] = Some(CardCosts { cards });
    }

    Ok(DoubleDummyPlay {
        opening_leader: play.opening_leader,
        tricks,
        end: play.end,
    })
}

/// The cards of `play` in the order they were played, up to the first one
/// not on record.
fn played_in_order(play: &PlaySequence) -> Vec<Card> {
    play.tricks
        .iter()
        .flat_map(|t| t.cards.iter())
        .map_while(|c| *c)
        .collect()
}

fn trump_of(strain: Strain) -> usize {
    match strain {
        Strain::Spades => SPADE,
        Strain::Hearts => HEART,
        Strain::Diamonds => DIAMOND,
        Strain::Clubs => CLUB,
        Strain::NoTrump => NOTRUMP,
    }
}

fn solver_card(card: Card) -> usize {
    let name = format!("{}{}", card.suit.to_char(), card.rank.to_char());
    // Every bridge_types card has a solver index
    parse_card(&name).unwrap_or(usize::MAX)
}

fn card_from_name(name: &str) -> Option<Card> {
    let mut chars = name.chars();
    let suit = Suit::from_char(chars.next()?)?;
    let rank = Rank::from_char(chars.next()?)?;
    Some(Card::new(suit, rank))
}

fn suit_order(suit: Suit) -> u8 {
    match suit {
        Suit::Spades => 0,
        Suit::Hearts => 1,
        Suit::Diamonds => 2,
        Suit::Clubs => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bridge_encodings::pbn::{double_dummy_play_to_pbn, read_pbn};

    /// Board 1 of a Brill daily IMP export: 3♦ by South, eight tricks.
    const RECORD: &str = r#"[Board "1"]
[Dealer "N"]
[Vulnerable "None"]
[Deal "N:QJ7.8.T9752.J752 K542.T7.QJ6.Q943 A863.Q932.K84.K8 T9.AKJ654.A3.AT6"]
[Declarer "S"]
[Contract "3D"]
[Result "8"]
[Play "W"]
HA H8 H7 H2
ST SJ S5 S3
DA D9 D6 D4
S9 SQ SK SA
H4 D2 HT H9
D3 D5 DJ DK
H5 D7 DQ H3
HJ S7 S4 S8
HK DT C4 HQ
CA C2 C3 CK
H6 C5 C9 D8
C6 CJ CQ C8
CT C7 S2 S6
"#;

    #[test]
    fn every_card_gets_its_alternatives() {
        let board = read_pbn(RECORD).unwrap().remove(0);
        let dd = analyse_board_play(&board, |_, _| true).unwrap();
        let play = board.play.as_ref().unwrap();
        assert_eq!(dd.tricks.len(), 13);
        for (t, trick) in play.tricks.iter().enumerate() {
            for slot in 0..4 {
                let card = trick.cards[slot].unwrap();
                let turn = dd.tricks[t].turns[slot].as_ref().unwrap();
                assert!(turn.cost_of(card).is_some(), "trick {t} slot {slot}");
                assert!(turn.cards.iter().any(|(_, c)| *c == Some(0)));
            }
        }
        // The section encodes against the same deal and play
        let lines = double_dummy_play_to_pbn(&dd, &board.deal, play).unwrap();
        assert_eq!(lines.len(), 13);
    }

    /// The played card's cost from its alternatives must be the cost the
    /// running trace finds from one solve per card: two routes to one number.
    #[test]
    fn agrees_with_the_running_trace() {
        use crate::analyse_play::{prefix_keys, running_trace};
        let board = read_pbn(RECORD).unwrap().remove(0);
        let dd = analyse_board_play(&board, |_, _| true).unwrap();
        let play = board.play.as_ref().unwrap();
        let played = played_in_order(play);
        let input = PlayInput {
            hands: Hands::from_pbn(&board.deal.to_pbn(Direction::North)).unwrap(),
            trump: DIAMOND,
            declarer: direction_to_seat(Direction::South),
            leader: direction_to_seat(Direction::West),
            plays: played.iter().map(|&c| solver_card(c)).collect(),
        };
        let keys = prefix_keys("deal", input.trump, input.leader, &input.plays);
        let trace = running_trace(&input, &keys, &Default::default()).unwrap();
        for (k, entry) in trace.trace.iter().enumerate() {
            let turn = dd.tricks[k / 4].turns[k % 4].as_ref().unwrap();
            assert_eq!(
                turn.cost_of(played[k]),
                Some(entry.cost),
                "card {k} {}",
                entry.card
            );
        }
    }

    #[test]
    fn stops_when_asked() {
        let board = read_pbn(RECORD).unwrap().remove(0);
        let result = analyse_board_play(&board, |done, _| done < 3);
        assert_eq!(result, Err(DdPlayError::Cancelled));
    }
}
