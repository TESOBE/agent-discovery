/// Integration test: Two agents discovering each other via loopback audio.

use agent_discovery::audio::device::LoopbackAudioEngine;
use agent_discovery::discovery::manager::{DiscoveryConfig, DiscoveryEvent, DiscoveryManager};
use agent_discovery::protocol::message::Capabilities;
use uuid::Uuid;

#[test]
fn test_two_agents_discover_each_other() {
    // All agents share the same frequency channel.
    // Self-echo is filtered by agent UUID.
    let agent1_id = Uuid::new_v4();
    let agent2_id = Uuid::new_v4();

    let config1 = DiscoveryConfig {
        agent_id: agent1_id,
        agent_address: "127.0.0.1:9001".to_string(),
        capabilities: Capabilities::new().with_tcp().with_claude(),
        ..Default::default()
    };

    let config2 = DiscoveryConfig {
        agent_id: agent2_id,
        agent_address: "127.0.0.1:9002".to_string(),
        capabilities: Capabilities::new().with_tcp().with_obp(),
        ..Default::default()
    };

    let (mut manager1, mut rx1) = DiscoveryManager::new(config1);
    let (mut manager2, mut rx2) = DiscoveryManager::new(config2);

    // Agent 1 sends an announce
    let announce1 = manager1.build_announce();
    let samples1 = manager1.encode_to_samples(&announce1);

    // Agent 2 receives and processes it
    let decoded = manager2
        .decode_from_samples(&samples1)
        .expect("Agent 2 should decode agent 1's announce");

    assert!(manager2.handle_message(&decoded));
    assert_eq!(manager2.peers().len(), 1);
    assert_eq!(manager2.peers()[0].agent_id, agent1_id);
    assert_eq!(manager2.peers()[0].address, "127.0.0.1:9001");

    // Agent 2 sends an announce
    let announce2 = manager2.build_announce();
    let samples2 = manager2.encode_to_samples(&announce2);

    // Agent 1 receives and processes it
    let decoded = manager1
        .decode_from_samples(&samples2)
        .expect("Agent 1 should decode agent 2's announce");

    assert!(manager1.handle_message(&decoded));
    assert_eq!(manager1.peers().len(), 1);
    assert_eq!(manager1.peers()[0].agent_id, agent2_id);
    assert_eq!(manager1.peers()[0].address, "127.0.0.1:9002");

    // Verify events
    match rx1.try_recv() {
        Ok(DiscoveryEvent::PeerDiscovered(peer)) => {
            assert_eq!(peer.agent_id, agent2_id);
        }
        other => panic!("Expected PeerDiscovered, got {:?}", other),
    }

    match rx2.try_recv() {
        Ok(DiscoveryEvent::PeerDiscovered(peer)) => {
            assert_eq!(peer.agent_id, agent1_id);
        }
        other => panic!("Expected PeerDiscovered, got {:?}", other),
    }
}

#[test]
fn test_loopback_engine_two_agents() {
    // Use a shared loopback engine to simulate audio between two agents
    let engine = LoopbackAudioEngine::new();

    let agent1_id = Uuid::new_v4();
    let agent2_id = Uuid::new_v4();

    let config1 = DiscoveryConfig {
        agent_id: agent1_id,
        agent_address: "127.0.0.1:9001".to_string(),
        capabilities: Capabilities::new().with_tcp(),
        ..Default::default()
    };

    let config2 = DiscoveryConfig {
        agent_id: agent2_id,
        agent_address: "127.0.0.1:9002".to_string(),
        capabilities: Capabilities::new().with_tcp(),
        ..Default::default()
    };

    let (manager1, _rx1) = DiscoveryManager::new(config1);
    let (mut manager2, _rx2) = DiscoveryManager::new(config2);

    // Agent 1 announces via loopback
    let announce = manager1.build_announce();
    let samples = manager1.encode_to_samples(&announce);
    engine.send_samples(samples).unwrap();

    // Agent 2 receives via loopback
    let received = engine.recv_samples().unwrap();
    let decoded = manager2
        .decode_from_samples(&received)
        .expect("Should decode from loopback");

    assert!(manager2.handle_message(&decoded));
    assert_eq!(manager2.peers().len(), 1);
}
