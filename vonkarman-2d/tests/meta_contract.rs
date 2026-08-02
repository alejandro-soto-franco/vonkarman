//! `export::Meta`'s field set is a contract the flowforms renderer (a separate
//! repository) codes against by key name. Nothing else in this workspace can
//! see that repository, so a rename or an added/removed field here would
//! break it silently. This test pins the exact key set.

use vonkarman_2d::export::Meta;

#[test]
fn meta_serialises_to_exactly_the_documented_keys() {
    let meta = Meta {
        nx: 256,
        ny: 128,
        lx: 24.0,
        ly: 12.0,
        u_mean: 1.0,
        radius: 0.5,
        cx: 6.0,
        cy: 6.0,
        dt: 1e-3,
        stride: 500,
        re: 100.0,
        frames: 5,
    };
    let value = serde_json::to_value(&meta).expect("serialise Meta");
    let object = value.as_object().expect("Meta serialises to a JSON object");

    let mut actual: Vec<&str> = object.keys().map(String::as_str).collect();
    actual.sort_unstable();

    let mut expected = vec![
        "nx", "ny", "lx", "ly", "u_mean", "radius", "cx", "cy", "dt", "stride", "re", "frames",
    ];
    expected.sort_unstable();

    assert_eq!(actual, expected, "Meta's JSON key set changed");
}
