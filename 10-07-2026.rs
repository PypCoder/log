use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

/// A simple inventory system simulating shared resource updates.
struct Inventory {
    // A Mutex is used here to safely allow interior mutability across threads,
    // protecting our HashMap from concurrent data races.
    items: Mutex<HashMap<String, u32>>,
}

fn main() {
    // 1. Initialize our thread-safe shared inventory.
    //    We wrap it in an Arc to permit shared, multi-owner reference counting.
    let shared_inventory = Arc::new(Inventory {
        items: Mutex::new(HashMap::new()),
    });

    let mut worker_threads = vec![];

    // Thread 1: Restock Gems, Potions, and Scrolls
    {
        let inventory_clone = Arc::clone(&shared_inventory);
        let handle = thread::spawn(move || {
            let mut data = inventory_clone.items.lock().unwrap();
            *data.entry("gems".to_string()).or_insert(0) += 3;
            *data.entry("potions".to_string()).or_insert(0) += 2;
            *data.entry("scrolls".to_string()).or_insert(0) += 1;
        });
        worker_threads.push(handle);
    }

    // Thread 2: Restock Gems, Potions, and Scrolls
    {
        let inventory_clone = Arc::clone(&shared_inventory);
        let handle = thread::spawn(move || {
            let mut data = inventory_clone.items.lock().unwrap();
            *data.entry("gems".to_string()).or_insert(0) += 4;
            *data.entry("potions".to_string()).or_insert(0) += 5;
            *data.entry("scrolls".to_string()).or_insert(0) += 2;
        });
        worker_threads.push(handle);
    }

    // 2. Wait for all workers to finish their restocking routines.
    for handle in worker_threads {
        handle.join().unwrap();
    }

    // 3. Print the final inventory. We sort the keys to guarantee consistent output.
    let final_data = shared_inventory.items.lock().unwrap();
    let mut sorted_inventory: Vec<(&String, &u32)> = final_data.iter().collect();
    sorted_inventory.sort_by(|a, b| a.0.cmp(b.0));

    for (item, quantity) in sorted_inventory {
        println!("{}: {}", item, quantity);
    }
}

/*
===============================================================================
THE COMPILER ERROR TEST BENCH (WHY RC BREAKS)
===============================================================================
Uncommenting the code below will trigger a static compilation error:
"error[E0277]: `Rc<RefCell<HashMap<String, u32>>>` cannot be sent between threads safely"

This is proof of Rust's static safety guards. The compiler stops compilation 
before a buggy program can ever be built or run on a production server.
===============================================================================

use std::rc::Rc;
use std::cell::RefCell;

fn compile_error_demo() {
    let unsafe_shared_inventory = Rc::new(RefCell::new(HashMap::<String, u32>::new()));
    
    let unsafe_clone = Rc::clone(&unsafe_shared_inventory);
    thread::spawn(move || {
        let mut data = unsafe_clone.borrow_mut();
        data.insert("unreachable_exploit".to_string(), 99);
    });
}
*/