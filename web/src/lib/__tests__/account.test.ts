import { describe, expect, it } from 'vitest'
import { bytes, initials, passwordStrength, safeNext } from '../account'

describe('initials', () => {
  it('takes the first and last word', () => {
    expect(initials('Zain Shahid')).toBe('ZS')
    expect(initials('  muhammad aamir gulzar ')).toBe('MG')
    expect(initials('Sana')).toBe('SA')
    expect(initials(' ')).toBe('?')
  })
})

describe('safeNext', () => {
  it('goes back only to a page of the studio', () => {
    expect(safeNext('/studio/sessions?session=3')).toBe('/studio/sessions?session=3')
    expect(safeNext('/studio')).toBe('/studio')
    // Anywhere else, another site above all, is not followed.
    for (const other of [null, '', '/', '//evil.example', 'https://evil.example/studio', '/studious', 'javascript:alert(1)']) {
      expect(safeNext(other)).toBe('/studio')
    }
  })
})

describe('passwordStrength', () => {
  it('goes by length first', () => {
    expect(passwordStrength('')).toBe(0)
    expect(passwordStrength('short')).toBe(1)
    expect(passwordStrength('aaaaaaaaaaaa')).toBe(1)
    expect(passwordStrength('abcdefgh')).toBe(2)
    expect(passwordStrength('correct horse')).toBeGreaterThanOrEqual(3)
    expect(passwordStrength('Correct horse battery 9')).toBe(4)
  })
})

describe('bytes', () => {
  it('reads as people say it', () => {
    expect(bytes(0)).toBe('0 B')
    expect(bytes(820 * 1024)).toBe('820 KB')
    expect(bytes(1.4 * 1024 * 1024)).toBe('1.4 MB')
    expect(bytes(24 * 1024 * 1024)).toBe('24 MB')
  })
})
