-- MySQL schema for BTTS app

CREATE DATABASE IF NOT EXISTS `btts_db` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `btts_db`;

-- Competitions
CREATE TABLE IF NOT EXISTS competitions (
  code VARCHAR(16) PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  country VARCHAR(128)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO competitions (code, name, country) VALUES
 ('PL','Premier League','England'),
 ('BL1','Bundesliga','Germany'),
 ('PD','La Liga','Spain'),
 ('SA','Serie A','Italy'),
 ('FL1','Ligue 1','France'),
 ('DED','Eredivisie','Netherlands'),
 ('PPL','Primeira Liga','Portugal'),
 ('CL','Champions League','Europe'),
 ('EL','Europa League','Europe'),
 ('ELC','Championship','England')
ON DUPLICATE KEY UPDATE name=VALUES(name), country=VALUES(country);

-- Teams
CREATE TABLE IF NOT EXISTS teams (
  id INT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  short_name VARCHAR(128),
  crest TEXT,
  competition_code VARCHAR(16),
  CONSTRAINT fk_team_comp FOREIGN KEY (competition_code) REFERENCES competitions(code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX teams_competition_idx ON `teams` (`competition_code`);
CREATE INDEX teams_name_idx ON `teams` (`name`);

-- Fixtures
CREATE TABLE IF NOT EXISTS fixtures (
  fixture_id INT PRIMARY KEY,
  competition_code VARCHAR(16),
  home_team_id INT,
  away_team_id INT,
  `utc_date` DATETIME,
  `status` VARCHAR(32) NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_fix_comp FOREIGN KEY (competition_code) REFERENCES competitions(code),
  CONSTRAINT fk_fix_home FOREIGN KEY (home_team_id) REFERENCES teams(id),
  CONSTRAINT fk_fix_away FOREIGN KEY (away_team_id) REFERENCES teams(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX fixtures_competition_idx ON `fixtures` (`competition_code`);
CREATE INDEX fixtures_date_idx ON `fixtures` (`utc_date`);

-- Team Statistics Cache
CREATE TABLE IF NOT EXISTS team_statistics (
  team_id INT NOT NULL,
  season VARCHAR(16) NOT NULL,
  stats JSON NOT NULL,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (team_id, season),
  CONSTRAINT fk_stats_team FOREIGN KEY (team_id) REFERENCES teams(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
  id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  fixture_id INT NULL,
  home_team_id INT NULL,
  away_team_id INT NULL,
  btts_probability DECIMAL(5,4) NOT NULL,
  btts_prediction TINYINT(1) NOT NULL,
  confidence DECIMAL(5,4) NOT NULL,
  model_type VARCHAR(64) NOT NULL,
  odds DECIMAL(10,3),
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT uq_predictions_fixture UNIQUE (fixture_id),
  CONSTRAINT fk_pred_home FOREIGN KEY (home_team_id) REFERENCES teams(id),
  CONSTRAINT fk_pred_away FOREIGN KEY (away_team_id) REFERENCES teams(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX predictions_created_at_idx ON `predictions` (`created_at`);
CREATE INDEX predictions_home_idx ON `predictions` (`home_team_id`);
CREATE INDEX predictions_away_idx ON `predictions` (`away_team_id`);

ALTER TABLE `predictions` ADD COLUMN `user_id` VARCHAR(64) NULL;
ALTER TABLE `predictions` DROP INDEX `uq_predictions_fixture`;
CREATE UNIQUE INDEX `uq_predictions_fixture_user` ON `predictions` (`fixture_id`, `user_id`);
CREATE INDEX `predictions_user_idx` ON `predictions` (`user_id`);

-- AI review caching on fixtures
ALTER TABLE `fixtures` ADD COLUMN `ai_review` TEXT NULL;
