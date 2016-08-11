-- ===========================
--         Temp File
-- ===========================
-- A temporary file (similar to the code below) will be created automatically
-- to set parameters and load this file

-- ** Parameters **
-- target = "111";          -- World Number - Level Number - Area Number
-- mode = "fast";           -- normal, fast, human
-- draw_tiles = "1";
-- meta = "0"               -- meta indicates multiple mission
-- pipe_name = "abc";

-- ** Loading main lua file **
-- f = assert (loadfile ("path_to_main_lua_file"));
-- f ();

-- ===========================
--         Parameters
-- ===========================
-- These parameters are expected to be generated and set in the temp file
-- Default values
mode = mode or "fast";
draw_tiles = tonumber(draw_tiles) or 0;
meta = tonumber(meta) or 0;
pipe_name = pipe_name or "";

-- Parsing world
if target then
    target_world = tonumber(string.sub(target, 1, 1));
    target_level = tonumber(string.sub(target, 2, 2));
    target_area = tonumber(string.sub(target, 3, 3));
else
    target = "111";
    target_world = 1;
    target_level = 1;
    target_area = 1;
end;

-- ===========================
--         Variables
-- ===========================
is_started = 0;             -- Indicates that the timer has started to decrease (i.e. commands can now be processed)
is_finished = 0;            -- Indicates a life has been lost, world has changed, or finish line crossed
last_time_left = 0;         -- Indicates the last time left (to check if timer has started to decrease)
skip_frames = 2;            -- Process a frame every 2 frames (usually 60 fps, by not returning 50% of the frames, we get ~30fps)
skip_screen = 0;            -- Does not send screen data to pipe (e.g. human mode)
skip_data = 0;              -- Does not send data to pipe (e.g. human mode)
skip_tiles = 0;             -- Does not send tiles to pipe (e.g. human mode)
skip_commands = 0;          -- Do not read commands from pipe (e.g. human mode)
start_delay = 100;          -- Number of frames to wait before pressing "start" to start level
send_all_pixels = 702;      -- Return full screen (all pixels) every 700 frames
force_refresh = 0;          -- Forces to return full screen (all pixels and data) for this number of frames
changing_level = 0;         -- Indicates level change in progress
curr_x_position = 0;        -- Current x position
curr_y_position = 0;        -- Current y position
last_processed_frame = 0;   -- Indicates last frame that was sent to pipe, to wait for commands on that frame number
commands = {};              -- List of current commands (inputs)
screen = {};                -- List of current screen pixels
data = {};                  -- List of current player stats
tiles = {};                 -- List of tiles
pipe_in = nil;              -- Input named pipe
pipe_out = nil;             -- Output named pipe
running_thread = 0;         -- To avoid 2 threads running at the same time
commands_rcvd = 0;          -- To indicate that commands were received

-- Max distances
distances = {};
distances["111"] = 3266;    -- 1-1
distances["123"] = 3266;    -- 1-2
distances["134"] = 2514;    -- 1-3
distances["145"] = 2430;    -- 1-4    
distances["211"] = 3298;    -- 2-1
distances["223"] = 3266;    -- 2-2
distances["234"] = 3682;    -- 2-3
distances["245"] = 2430;    -- 2-4    
distances["311"] = 3298;    -- 3-1
distances["322"] = 3442;    -- 3-2    
distances["333"] = 2498;    -- 3-3
distances["344"] = 2430;    -- 3-4
distances["411"] = 3698;    -- 4-1
distances["423"] = 3266;    -- 4-2
distances["434"] = 2434;    -- 4-3
distances["445"] = 2942;    -- 4-4
distances["511"] = 3282;    -- 5-1
distances["522"] = 3298;    -- 5-2
distances["533"] = 2514;    -- 5-3
distances["544"] = 2429;    -- 5-4
distances["611"] = 3106;    -- 6-1
distances["622"] = 3554;    -- 6-2
distances["633"] = 2754;    -- 6-3
distances["644"] = 2429;    -- 6-4
distances["711"] = 2962;    -- 7-1
distances["723"] = 3266;    -- 7-2
distances["734"] = 3682;    -- 7-3
distances["745"] = 3453;    -- 7-4
distances["811"] = 6114;    -- 8-1
distances["822"] = 3554;    -- 8-2
distances["833"] = 3554;    -- 8-3
distances["844"] = 4989;    -- 8-4

-- Setting mode
-- Human: Game is played manually by a human (no algo)
-- Normal: Game is played by algo at normal speed so human can watch
-- Fast (Default): Game is played by algo at high speed, hard for human to watch
if mode == "human" then
    emu.speedmode("normal");
    skip_frames = 1;
    skip_screen = 1;
    skip_data = 1;
    skip_tiles = 1;
    skip_commands = 1;
    start_delay = 125;

elseif mode == "normal" then
    emu.speedmode("normal");
    skip_frames = 2;
    start_delay = 100;

else
    -- Fast
    emu.speedmode("maximum");
    skip_frames = 2;
    start_delay = 175;
    send_all_pixels = 1500;
end;

-- ===========================
--         Memory Address
-- ===========================
addr_world = 0x075f;
addr_level = 0x075c;
addr_area = 0x0760;
addr_life = 0x075a;
addr_score = 0x07de;
addr_time = 0x07f8;
addr_coins = 0x07ed;
addr_curr_page = 0x6d;
addr_curr_x = 0x86;
addr_curr_y = 0x03b8;
addr_left_x = 0x071c;
addr_player_state = 0x000e;     -- x06 dies, x0b dying
addr_player_status = 0x0756;    -- 0 = small, 1 = big, 2+ = fiery
addr_enemy_page = 0x6e;
addr_enemy_x = 0x87;
addr_enemy_y = 0xcf;
addr_injury_timer = 0x079e;
addr_swimming_flag = 0x0704;
addr_tiles = 0x500;

-- ===========================
--         Functions
-- ===========================
-- Initiating variables
function reset_vars()
    for x=0,255 do
        screen[x] = {};
        for y=0,223 do
            screen[x][y] = -1;
        end;
    end;
    local data_var = { "distance", "life", "score", "coins", "time", "player_status", "is_finished" };
    for i=1,#data_var do
        data[data_var[i]] = -1;
    end;
    local commands_var = { "up", "left", "down", "right", "A", "B", "start", "select" };
    for i=1,#commands_var do
        commands[commands_var[i]] = false;
    end;
    for x=0,15 do
        tiles[x] = {};
        for y=0,12 do
            tiles[x][y] = -1;
        end;
    end;
    is_started = 0;
    is_finished = 0;
    last_time_left = 0;
    curr_x_position = 0;
    curr_y_position = 0;
    last_processed_frame = 0;
    max_distance = distances[target] or 0;
end;

-- round - Rounds a number to precision level
function round(number, precision)
  local mult = 10^(precision or 0);
  return math.floor(number * mult + 0.5) / mult;
end;

-- split - Splits a string with a specific delimiter
function split(self, delimiter)
    local results = {};
    local start = 1;
    local split_start, split_end  = string.find(self, delimiter, start);
    while split_start do
        table.insert(results, string.sub(self, start, split_start - 1));
        start = split_end + 1;
        split_start, split_end = string.find(self, delimiter, start);
    end;
    table.insert(results, string.sub(self, start));
    return results;
end;

-- readbyterange - Reads a range of bytes and return a number
function readbyterange(address, length)
  local return_value = 0;
  for offset = 0,length-1 do
    return_value = return_value * 10;
    return_value = return_value + memory.readbyte(address + offset);
  end;
  return return_value;
end

-- get_level - Returns current level (0-indexed) (0 to 31)
function get_level()
    return memory.readbyte(addr_world) * 4 + memory.readbyte(addr_level);
end;

-- get_world_number - Returns current world number (1 to 8)
function get_world_number()
    return memory.readbyte(addr_world) + 1;
end;

-- get_level_number - Returns current level number (1 to 4)
function get_level_number()
    return memory.readbyte(addr_level) + 1;
end;

-- get_area_number - Returns current area number (1 to 5)
function get_area_number()
    return memory.readbyte(addr_area) + 1;
end;

-- get_coins - Returns the number of coins collected (0 to 99)
function get_coins()
    return tonumber(readbyterange(addr_coins, 2));
end;

-- get_life - Returns the number of remaining lives
function get_life()
    return memory.readbyte(addr_life) + 1;
end;

-- get_score - Returns the current player score (0 to 999990)
function get_score()
    return tonumber(readbyterange(addr_score, 6));
end;

-- get_time - Returns the time left (0 to 999)
function get_time()
    return tonumber(readbyterange(addr_time, 3));
end;

-- get_x_position - Returns the current (horizontal) position
function get_x_position()
    return memory.readbyte(addr_curr_page) * 0x100 + memory.readbyte(addr_curr_x);
end;

-- get_left_x_position - Returns number of pixels from left of screen
function get_left_x_position()
    return (memory.readbyte(addr_curr_x) - memory.readbyte(addr_left_x)) % 256;
end;

-- get_y_position - Returns the current (vertical) position
function get_y_position()
    return memory.readbyte(addr_curr_y);
end;

-- update_positions - Update x and y position variables
function update_positions()
    curr_x_position = get_x_position();
    curr_y_position = get_y_position();
    return;
end;

-- get_is_dead - Returns 1 if the player is dead or dying
-- 0x06 means dead, 0x0b means dying
function get_is_dead()
    local player_state = memory.readbyte(addr_player_state);
    if (player_state == 0x06) or (player_state == 0x0b) then
        return 1;
    else
        return 0;
    end;
end;

-- get_player_status - Returns the player status
-- 0 is small, 1 is big, 2+ is fiery (can shoot fireballs)
function get_player_status()
    return memory.readbyte(addr_player_status);
end;

-- get_tile_type - Returns the tile type
-- 0 = empty space, 1 = non-empty space
function get_tile_type(box_x, box_y)
    local left_x = get_left_x_position();
    local x = curr_x_position - left_x + box_x + 112;
    local y = box_y + 96;
    local page = math.floor(x / 256) % 2;
    
    local sub_x = math.floor((x % 256) / 16);
    local sub_y = math.floor((y - 32) / 16);
    local curr_tile_addr = addr_tiles + page * 13 * 16 + sub_y * 16 + sub_x;
    
    if (sub_y >= 13) or (sub_y < 0) then
      return 0;
    end;
    
    -- 0 = empty space, 1 is not-empty (e.g. hard surface or object)
    if memory.readbyte(curr_tile_addr) ~= 0 then
      return 1;
    else
      return 0;
    end;
end;

-- get_enemies - Returns enemy location
function get_enemies()
    local enemies = {};
    for slot=0,4 do
      local enemy = memory.readbyte(0xF + slot);
      if enemy ~= 0 then
        local ex = memory.readbyte(addr_enemy_page + slot) * 0x100 + memory.readbyte(addr_enemy_x + slot);
        local ey = memory.readbyte(addr_enemy_y + slot);
        enemies[#enemies+1] = {["x"]=ex,["y"]=ey};
      end
    end
    return enemies;
end;

-- get_distance_perc - Returns the percentage of the world currently completed (as a string with % sign)
function get_distance_perc(current_distance, max_distance)
    -- For some maps, underground tunnels use a page location after the finish line
    -- Not returning a percentage for those cases, or if max_distance is 0
    if (current_distance > (max_distance + 40)) or (max_distance <= 0) then
        return "--%";
    end;
    -- There are usually 80 pixels between the flagpole and the castle.
    -- The target (reward_threshold) is 40 pixels before the castle
    -- The finish line (where the game will automatically close) is 15 pixels before the castle
    local current_perc = round(current_distance / (max_distance - 40) * 100, 0);
    current_perc = math.min(current_perc, 100);
    current_perc = math.max(current_perc, 0);
    return current_perc .. "%";
end;

-- show_curr_distance - Displays the current distance on the map with percentage
function show_curr_distance()
    local distance = "Distance " .. curr_x_position;
    distance = distance .. " (" .. get_distance_perc(curr_x_position, max_distance) .. ")";
    return emu.message(distance);
end;

-- get_data - Returns the current player stats data (reward, distance, life, scores, coins, timer, player_status, is_finished)
-- Only returning values that have changed since last update
-- Format: data_<frame_number>#name_1:value_1|name_2:value_2|name_3:value_3
function get_data()

    -- Skipping data is skip_data is set
    if skip_data == 1 then
        return;
    end;

    local framecount = emu.framecount();
    local data_count = 0;
    local data_string = "";
    local curr_life = get_life();
    local curr_score = get_score();
    local curr_coins = get_coins();
    local curr_time = get_time();
    local curr_player_status = get_player_status();
    
    -- Checking what values have changed
    if (framecount % send_all_pixels == 0) or (curr_x_position ~= data["distance"]) or (force_refresh > 0) then
        data["distance"] = curr_x_position;
        data_string = data_string .. "|distance:" .. curr_x_position;
        data_count = data_count + 2;
    end;
    if (framecount % send_all_pixels == 0) or (curr_life ~= data["life"]) or (force_refresh > 0) then
        data["life"] = curr_life;
        data_string = data_string .. "|life:" .. curr_life;
        data_count = data_count + 1;
    end;
    if (framecount % send_all_pixels == 0) or (curr_score ~= data["score"]) or (force_refresh > 0) then
        data["score"] = curr_score;
        data_string = data_string .. "|score:" .. curr_score;
        data_count = data_count + 1;
    end;
    if (framecount % send_all_pixels == 0) or (curr_coins ~= data["coins"]) or (force_refresh > 0) then
        data["coins"] = curr_coins;
        data_string = data_string .. "|coins:" .. curr_coins;
        data_count = data_count + 1;
    end;
    if (framecount % send_all_pixels == 0) or (curr_time ~= data["time"]) or (force_refresh > 0) then
        data["time"] = curr_time;
        data_string = data_string .. "|time:" .. curr_time;
        data_count = data_count + 1;
    end;
    if (framecount % send_all_pixels == 0) or (curr_player_status ~= data["player_status"]) or (force_refresh > 0) then
        data["player_status"] = curr_player_status;
        data_string = data_string .. "|player_status:" .. curr_player_status;
        data_count = data_count + 1;
    end;
    if (framecount % send_all_pixels == 0) or (is_finished ~= data["is_finished"]) or (force_refresh > 0) then
        data["is_finished"] = is_finished;
        data_string = data_string .. "|is_finished:" .. is_finished;
        data_count = data_count + 1;
    end;
    
    -- Removing leading "|" if data has changed, otherwise not returning anything
    if data_count > 0 then
        if is_finished == 1 then
            -- Indicates to the listening thread to also exit after parsing command
            data_string = data_string .. "|exit";
        end;
        write_to_pipe("data_" .. framecount .. "#" .. string.sub(data_string, 2, -1));
    end;
    return;
end;

-- get_screen - Returns the current RGB data for the screen (256 x 224)
-- Only returns pixels that have changed since last frame update
-- Format: screen_<frame_number>#<x(2 hex digits)><y (2 hex digits)><palette (2 hex digits)>|...
-- Palette is a number from 0 to 127 that represents an RGB color (conversion table in python file)
function get_screen()

    -- Skipping screen is skip_screen is set
    if skip_screen == 1 then
        return;
    end;

    local r, g, b, p;
    local framecount = emu.framecount();
    -- NES only has y values in the range 8 to 231, so we need to offset y values by 8
    local offset_y = 8;
    for y=0,223 do
        local screen_string = "";
        local data_count = 0;
        for x=0,255 do
            r, g, b, p = emu.getscreenpixel(x, y + offset_y, false);
            if (framecount % send_all_pixels == 0) or (p ~= screen[x][y]) or (force_refresh > 0) then
                screen[x][y] = p;
                screen_string = screen_string .. "|" .. string.format("%02x%02x%02x", x, y, p);
                data_count = data_count + 1;
            end;
        end;
        if data_count > 0 then
            write_to_pipe("screen_" .. framecount .. "#" .. string.sub(screen_string, 2, -1));
        end;
    end;
    return;
end;

-- get_tiles - Returns tiles data (and displays them on screen)
-- Only returns tiles that have changed since last update
-- Format: tiles_<frame_number>#<x(1 hex digits)><y (1 hex digits)><value (1 hex digits)>|...
-- Value: 0 - Empty space, 1 - Object / Other, 2 - Enemy, 3 - Mario
function get_tiles()
    
    -- Skipping if we do not need to draw tiles
    if draw_tiles == 0 then
        return;
    end;
    
    local enemies = get_enemies();
    local left_x = get_left_x_position();
    local framecount = emu.framecount();
    
    -- Outside box (80 x 65 px)
    -- Will contain a matrix of 16x13 sub-boxes of 5x5 pixels each
    gui.box(
        50 - 5 * 7 - 2,
        70 - 5 * 7 - 2,
        50 + 5 * 8 + 3,
        70 + 5 * 5 + 3,
        0,
        "P30"
    );       -- P30 = White (NES Palette 30 color)
  
    -- Calculating tile types
    for box_y = -4*16,8*16,16 do
        local tile_string = "";
        local data_count = 0;
        for box_x = -7*16,8*16,16 do
      
            -- 0 = Empty space
            local tile_value = 0;
            local color = 0;
            local fill = 0;
      
            -- +1 = Not-Empty space (e.g. hard surface, object)
            local curr_tile_type = get_tile_type(box_x, box_y);
            if (curr_tile_type == 1) and (curr_y_position + box_y < 0x1B0) then
                tile_value = 1;
                color = "P30"; -- White (NES Palette 30 color)
            end;
      
            -- +2 = Enemies
            for i = 1,#enemies do
                local dist_x = math.abs(enemies[i]["x"] - (curr_x_position + box_x - left_x + 108));
                local dist_y = math.abs(enemies[i]["y"] - (90 + box_y));
                if (dist_x <= 8) and (dist_y <= 8) then
                    tile_value = 2;
                    color = "P27"; -- Orange (NES Palette 27 color)
                    fill = "P3F"; -- Black (NES Palette 3F color);
                end;
            end;
            
            -- +3 = Mario
            local dist_x = math.abs(curr_x_position - (curr_x_position + box_x - left_x + 108));
            local dist_y = math.abs(curr_y_position - (80 + box_y));
            if (dist_x <= 8) and (dist_y <= 8) then
                tile_value = 3;
                color = "P05"; -- Red (NES Palette 05 color)
                fill = color;
            end;
            
            -- Drawing tile
            local tile_x = 50 + 5 * (box_x / 16);
            local tile_y = 55 + 5 * (box_y / 16);
            
            if (tile_value ~= 0) then
                gui.box(tile_x - 2, tile_y - 2, tile_x + 2, tile_y + 2, fill, color);
            end;
            
            -- Storing value only on processed frames
            -- Only sending values for processed frames if skip_tiles is 0
            -- Skipped frames (where commands are not processed) have box drawn, but no values sent
            if (framecount % skip_frames == 0) and (skip_tiles == 0) then
                -- Only returning value if tile value has changed (or full refresh needed)
                if (framecount % send_all_pixels == 0) or (tile_value ~= tiles[(box_x / 16) + 7][(box_y / 16) + 4]) or (force_refresh > 0) then
                    tiles[(box_x / 16) + 7][(box_y / 16) + 4] = tile_value;
                    --noinspection StringConcatenationInLoops
                    tile_string = tile_string .. "|" .. string.format("%01x%01x%01x", (box_x / 16) + 7, (box_y / 16) + 4, tile_value);
                    data_count = data_count + 1;
                end;
            end;
        end;
        if data_count > 0 then
            write_to_pipe("tiles_" .. framecount .. "#" .. string.sub(tile_string, 2, -1));
        end;
    end;
    return;
end;

-- check_if_started - Checks if the timer has started to decrease
-- this is to avoid receiving commands while the level is loading, or the animation is still running
function check_if_started()
    local time_left = get_time();

    -- Cannot start before 'start' is pressed
    local framecount = emu.framecount();
    if (framecount < start_delay) then
        return;
    end;

    -- Checking if time has decreased
    if (time_left > 0) and (is_finished ~= 1) then
        -- Level started (if timer decreased)
        if (last_time_left > time_left) then
            is_started = 1;
            last_time_left = 0;
            pipe_out, _, _ = io.open("/tmp/smb-fifo-in." .. pipe_name, "w");
            write_to_pipe("ready_" .. emu.framecount());
            force_refresh = 5;  -- Sending full screen for next 5 frames, then only diffs
            update_positions();
            show_curr_distance();
            get_tiles();
            get_data();
            -- get_screen();    -- Was blocking execution
            ask_for_commands();
        else
            last_time_left = time_left;
        end;
    end;
    return;
end;

-- check_if_finished - Checks if the level is finished (life lost, finish line crossed, level increased)
-- The target (reward_threshold) is 40 pixels before the castle
-- The finish line (where the game will automatically close) is 15 pixels before the castle
function check_if_finished()
    if (get_is_dead() == 1)
        or ((curr_x_position >= max_distance - 15) and (curr_x_position <= max_distance))
        or (get_life() < 3)
        or (get_level() > 4 * (target_world - 1) + (target_level - 1)) then
        -- Level finished
        -- is_finished will be written to pipe with the get_data() function
        is_started = 0;
        is_finished = 1;

        -- Processing manually last command
        read_commands();
        if commands_rcvd == 1 then
            commands_rcvd = 0
            emu.frameadvance();
            update_positions();
            show_curr_distance();
            get_tiles();
            get_data();
            get_screen();
            ask_for_commands();
        end;
    end;
    return;
end;

-- ask_for_commands - Mark the current frame has processed (to listen for matching command)
function ask_for_commands()
    local framecount = emu.framecount();
    last_processed_frame = framecount;
    write_to_pipe("done_" .. framecount);
end;

-- receive_commands() - Wait for commands in input pipe
function read_commands()
    -- Cant read if pipe_in is not set
    if not pipe_in then
        return;
    end;
    
    -- Waiting for proper line
    local is_received = 0;
    local line = "";
    local line = pipe_in:read();
    if line ~= nil then
        parse_commands(line);
    end;
    return;
end;

-- parse_commands() - Parse received commands
-- Format: commands_<frame number>#up,left,down,right,a,b (e.g. commands_21345#0,0,0,1,1,0)
-- Format: changelevel#<level_number> (e.g. changelevel#22) (level number is a number from 0 to 31)
-- Format: exit
function parse_commands(line)
    -- Splitting line
    local parts = split(line, "#");
    local header = parts[1] or "";
    local data = parts[2] or "";
    parts = split(header, "_");
    local command = parts[1] or "";
    local frame_number = parts[2] or "";

    -- Deciding what command to execute
    -- Setting joypad
    if ("commands" == command) and (tonumber(frame_number) == last_processed_frame) then
        commands_rcvd = 1;
        parts = split(data, ",");
        commands["up"] = ((parts[1] == "1") or (parts[1] == "true"));
        commands["left"] = ((parts[2] == "1") or (parts[2] == "true"));
        commands["down"] = ((parts[3] == "1") or (parts[3] == "true"));
        commands["right"] = ((parts[4] == "1") or (parts[4] == "true"));
        commands["A"] = ((parts[5] == "1") or (parts[5] == "true"));
        commands["B"] = ((parts[6] == "1") or (parts[6] == "true"));
        commands["start"] = false;
        commands["select"] = false;
        joypad.set(1, commands);

    -- Noop at beginning of level (to simulate seed)
    elseif ("noop" == command) and (tonumber(frame_number) == last_processed_frame) then
        local noop_count = tonumber(data);
        commands["up"] = false;
        commands["left"] = false;
        commands["down"] = false;
        commands["right"] = false;
        commands["A"] = false;
        commands["B"] = false;
        commands["start"] = false;
        commands["select"] = false;
        joypad.set(1, commands);
        if noop_count > 0 then
            for i=1,noop_count,1 do
                emu.frameadvance();
            end;
        end;

    -- Changing level
    elseif ("changelevel" == command) and (tonumber(data) >= 0) and (tonumber(data) <= 31) then
        local level = tonumber(data)
        target_world = math.floor(level / 4) + 1
        target_level = (level % 4) + 1
        target_area = target_level
        if (target_world == 1) or (target_world == 2) or (target_world == 4) or (target_world == 7) then
            if (target_level >= 2) then
                target_area = target_area + 1;
            end;
        end;
        target = target_world .. target_level .. target_area;
        is_started = 0;
        is_finished = 0;
        changing_level = 0;
        reset_vars();
        emu.softreset();

    -- Exiting
    elseif "exit" == command then
        close_pipes();
        os.exit()
    end;
    return;
end;

-- open_pipes - Open required pipes to inter-process communication
-- pipes (mkfifo) are created by python script
function open_pipes()
    local _;
    if pipe_name ~= "" and mode ~= "human" then
        pipe_out, _, _ = io.open("/tmp/smb-fifo-in." .. pipe_name, "w");
        pipe_in, _, _ = io.open("/tmp/smb-fifo-out." .. pipe_name, "r");
    end;
    return;
end;

-- close_pipes - Close pipes before exiting
-- pipes (mkfifo) are created by python script
function close_pipes()
    if pipe_in then
        pipe_in:close();
    end;
    if pipe_out then
        pipe_out:close();
    end;
    return;
end;

-- write_to_pipe - Write data to pipe
function write_to_pipe(data)
    if data and pipe_out then
        pipe_out:write(data .. "!\n");
        pipe_out:flush();
    end;
    return;
end;

-- ===========================
--         Hooks
-- ===========================
-- Hook to change level on load
function hook_set_world()
    if (get_world_number() ~= target_world) then
        memory.writebyte(addr_world, (target_world - 1));
        memory.writebyte(addr_level, (target_level - 1));
        memory.writebyte(addr_area, (target_area - 1));
    end;
end;
function hook_set_level()
    if (get_level_number() ~= target_level) then
        memory.writebyte(addr_world, (target_world - 1));
        memory.writebyte(addr_level, (target_level - 1));
        memory.writebyte(addr_area, (target_area - 1));
    end;
end;
function hook_set_area()
    if (get_area_number() ~= target_area) then
        memory.writebyte(addr_world, (target_world - 1));
        memory.writebyte(addr_level, (target_level - 1));
        memory.writebyte(addr_area, (target_area - 1));
    end;
end;
memory.registerwrite(addr_world, hook_set_world);
memory.registerwrite(addr_level, hook_set_level);
memory.registerwrite(addr_area, hook_set_area);

function exit_hook()
    write_to_pipe("exit");
    close_pipes();
end;
emu.registerexit(exit_hook);

-- ===========================
--      ** DEBUG **
-- ===========================
-- Functions used to debug levels (you will be an invincible swimmer with unlimited lives)
-- function hook_set_life()
--     if memory.readbyte(addr_life) ~= 0x08 then
--         memory.writebyte(addr_life, 0x08);
--     end;
-- end;
-- memory.registerwrite(addr_life, hook_set_life);
-- 
-- function hook_set_invincibility()
--     if memory.readbyte(addr_injury_timer) ~= 0x08 then
--         memory.writebyte(addr_injury_timer, 0x08);
--     end;
-- end;
-- memory.registerwrite(addr_injury_timer, hook_set_invincibility);
-- 
-- function hook_set_swimmer()
--     if memory.readbyte(addr_swimming_flag) ~= 0x01 then
--         memory.writebyte(addr_swimming_flag, 0x01);
--     end;
-- end;
-- memory.registerwrite(addr_swimming_flag, hook_set_swimmer);

-- ===========================
--         Main Loop
-- ===========================
-- Opening pipes
reset_vars();
open_pipes();

function main_loop()
    if running_thread == 1 then
        return;
    end;
    running_thread = 1;
    local framecount = emu.framecount();

    -- Checking if game is started or is finished
    if is_started == 0 then
        check_if_started();
    elseif is_finished == 0 then
        check_if_finished();
    end;

    -- Checking if game has started, if not, pressing "start" to start it
    if (0 == is_started) and (framecount == start_delay) then
        commands["start"] = true;
        joypad.set(1, commands);
        emu.frameadvance();
        commands["start"] = false;

    -- Game not yet started, just skipping frame
    elseif 0 == is_started then
        emu.frameadvance();

    -- Human mode
    elseif 'human' == mode then
        emu.frameadvance();
        update_positions();
        show_curr_distance();
        get_tiles();

    -- Processed frame, getting commands (sync mode), sending back screen
    elseif framecount % skip_frames == 0 then
        read_commands();
        if commands_rcvd == 1 then
            commands_rcvd = 0
            emu.frameadvance();
            update_positions();
            show_curr_distance();
            get_tiles();
            get_data();
            get_screen();
            ask_for_commands();
        end;

    -- Skipped frame, using same command as last frame, not returning screen
    else
        joypad.set(1, commands);
        emu.frameadvance();
        update_positions();
        show_curr_distance();
        get_tiles();
    end;

    -- Exiting if game is finished
    if (1 == is_finished) then
        if 0 == meta then
            -- Single Mission
            for i=1,20,1 do         -- Gives python a couple of ms to process it
                emu.frameadvance();
            end
            write_to_pipe("exit");  -- Sends exit
            close_pipes();
            os.exit();

        elseif 0 == changing_level then
            -- Meta mission - Sending level change required
            get_data();              -- Sends is_finished
            write_to_pipe("reset");  -- Tells python to reset frame number and send change level
            changing_level = 1;
        else
            -- Waiting for level change
            read_commands();
        end;

    end;
    force_refresh = force_refresh - 1;
    if force_refresh < 0 then force_refresh = 0; end;
    running_thread = 0;
end;

while (true) do
    main_loop();
end;
